#!/usr/bin/env python
"""
Revised validation script with GPU acceleration and vectorized batch processing.
Key improvements:
  - Removed duplicate imports and forced CPU settings.
  - Loads model once and uses model.predict on an entire batch.
  - Vectorizes flux normalization and (where possible) redshift computations.
  - Retains the original logic for computing Gaussian templates and best shifts.
  - Saves results incrementally to an HDF5 file.
"""

import os
import sys
import math
import time
import json
import yaml
import h5py
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

from keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Lambda, MultiHeadAttention, LayerNormalization, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant, Zeros

import astropy.io.fits as pyfits
from itertools import chain

# -------------------------------
# Remove any forced CPU settings!
# (Comment out or delete any os.environ that forces CPU)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set temporary directories for matplotlib if needed
os.environ['MPLCONFIGDIR'] = 'tmp/matplotlib'
os.environ['TMPDIR'] = 'tmp'
os.makedirs('tmp/matplotlib', exist_ok=True)
os.makedirs('tmp', exist_ok=True)

# -------------------------------
# Global variables (will be set by configuration)
global len_data, max_z, wavelength_template, means, columns, validation_config

# -------------------------------
# Load configuration file and set global parameters
config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

physics_info = config['physicsinfo']
validation_config = config['validationinfo']

max_z = physics_info['max_z']  # maximum expected redshift
len_data = physics_info['len_data']  # number of wavelength bins
x_max = np.log10(physics_info['x_max'])  # maximum observed wavelength (in log10 Angstroms)
resolution = physics_info['resolution']  # resolution in log10 Angstroms

# Compute the number of pixels in the wavelength template 
num_pixels = len_data + math.ceil(np.log10(max_z+1) / resolution)
wavelength_template = np.arange(x_max - resolution * num_pixels, x_max, resolution)

# Emission/absorption lines (assumes physics_info['means'] is provided)
means = np.log10(physics_info['means'])

# Optionally, fix column names (if some are list types)
def fix_names_list(names):
    fixed_names = []
    for name in names:
        if isinstance(name, list):
            fixed_names.append("[" + ", ".join(name) + "]")
        else:
            fixed_names.append(name)
    return fixed_names

columns = fix_names_list(physics_info['columns'])

# -------------------------------
# Custom layers and functions used by the model

def inverted_relu(x):
    return -tf.nn.relu(x)

class CustomMatrixMultiplication(Layer):
    def call(self, inputs, **kwargs):
        matrix1, matrix2 = inputs
        matrix1_transposed = tf.transpose(matrix1, perm=[0, 2, 1])
        matrix2_transposed = tf.transpose(matrix2, perm=[0, 2, 1])
        result = tf.matmul(matrix1_transposed, matrix2_transposed)
        return tf.reduce_sum(result, axis=-1)

class ScaledSigmoid(Layer):
    def __init__(self, min_val, max_val, steepness=0.1, **kwargs):
        super(ScaledSigmoid, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.steepness = steepness

    def call(self, inputs, **kwargs):
        sigmoid = tf.nn.sigmoid(inputs * self.steepness)
        return self.min_val + (self.max_val - self.min_val) * sigmoid

    def get_config(self):
        config = super(ScaledSigmoid, self).get_config()
        config.update({'min_val': self.min_val, 'max_val': self.max_val, 'steepness': self.steepness})
        return config

class GlobalSumPooling1D(Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

# Custom objects for model loading
custom_objects = {
    'ScaledSigmoid': ScaledSigmoid,
    'GlobalSumPooling1D': GlobalSumPooling1D,
    'inverted_relu': inverted_relu
}

# -------------------------------
# Helper functions for redshift and flux template conversion

def redshift_to_shift(z, wavelength_template):
    # Works on a single redshift value
    x = wavelength_template
    obs_wavelength_log = x[-1]
    delta_log = np.log10(1 + z)
    em_wavelength_log = obs_wavelength_log - delta_log
    shift = np.argmin(np.abs(x - em_wavelength_log))
    return shift - len_data + 1

def shift_to_redshift(shift, wavelength_template):
    x = wavelength_template
    obs_wavelength_log = x[-1]
    em_wavelength_log = x[shift + len_data - 1]
    delta_log = obs_wavelength_log - em_wavelength_log
    return 10 ** delta_log - 1

@tf.function
def compute_batch_gaussians_tf(template, batch_amplitudes, batch_std_devs):
    # Convert template from log10 scale back to linear
    template = tf.cast(10 ** template, dtype=tf.float32)
    batch_amplitudes = tf.cast(batch_amplitudes, dtype=tf.float32)
    batch_std_devs = tf.cast(batch_std_devs + 5e-7, dtype=tf.float32)
    means_cbg = tf.constant(10 ** means, dtype=tf.float32)

    std_dev_expanded = tf.reshape(batch_std_devs, (-1, 1, 1))
    expanded_template = tf.expand_dims(template, 1)  # shape [N, 1]
    expanded_means = tf.expand_dims(means_cbg, 0)      # shape [1, M]
    gaussians = (1 / (std_dev_expanded * tf.math.sqrt(2 * math.pi))) * \
                tf.exp(-0.5 * tf.square((expanded_template - expanded_means) / std_dev_expanded))
    gaussians = tf.transpose(gaussians, perm=[0, 2, 1])
    batch_amplitudes_expanded = tf.expand_dims(batch_amplitudes, 2)
    gaussians_scaled = gaussians * batch_amplitudes_expanded
    summed_gaussians = tf.reduce_sum(gaussians_scaled, axis=1)
    return summed_gaussians

@tf.function
def find_min_euclidean_distance_index(large_arrays, tiny_arrays, alpha=0.9, k=1900, radius=1.0):
    # Ensure data types are consistent
    large_arrays = tf.cast(large_arrays, dtype=tf.float32)
    tiny_arrays = tf.cast(tiny_arrays, dtype=tf.float32)

    # Dimensions of the inputs
    batch_size = tf.shape(large_arrays)[0]
    large_length = tf.shape(large_arrays)[1]
    tiny_length = tf.shape(tiny_arrays)[1]
    # Determine the number of sliding windows possible
    num_windows = large_length - tiny_length + 1

    # Create indices for all windows
    indices = tf.expand_dims(tf.range(num_windows), 0) + tf.expand_dims(tf.range(tiny_length), 1)

    # Batch and tile indices to gather windows across the batch
    indices = tf.tile(indices[None, :, :], [batch_size, 1, 1])

    # Gather windows from the large arrays
    large_windows = tf.gather(large_arrays, indices, batch_dims=1)

    # Compute squared differences and mean over the tiny_length dimension to get the MSE
    squared_diff = tf.square(large_windows - tiny_arrays[:, :, None])
    mse = tf.reduce_mean(squared_diff, axis=1)

    # Compute dot products and cosine similarities
    dot_products = tf.reduce_sum(tf.multiply(large_windows, tiny_arrays[:, :, None]), axis=1)
    norm_large = tf.norm(large_windows, axis=1)
    norm_tiny = tf.norm(tiny_arrays, axis=1, keepdims=True)
    cosine_similarities = dot_products / (norm_large * norm_tiny)

    # Hybrid loss calculation
    hybrid_loss = alpha * mse + (1 - alpha) * -cosine_similarities  # Maximizing cosine similarity is minimizing its negative
    # Find the indices of the top k smallest hybrid loss values in each batch
    values, indices = tf.nn.top_k(-hybrid_loss, k, sorted=True)
    values = -values  # Convert back to positive values

    # Generate exponentially decaying weights based on the radius
    weights = tf.exp(-tf.range(k, dtype=tf.float32) / radius)
    weights /= tf.reduce_sum(weights)  # Normalize weights to sum to 1

    # Calculate the weighted average of these top k values
    weighted_top_k_avg = tf.reduce_sum(values * weights, axis=1)

    # Calculate the average of these weighted top k values
    loss = tf.reduce_mean(weighted_top_k_avg)

    # Return the indices corresponding to the smallest hybrid loss (i.e., best matches)
    best_match_indices = tf.argmin(hybrid_loss, axis=1)

    return best_match_indices, loss, hybrid_loss


def width_to_velocity(width):
    center = 6500
    return ((width) / center) * 300000

# -------------------------------
# Preprocessing functions remain as originally defined

def preprocess_flux(flux, variance, threshold, wavelengths=None, skylines=None):
    flux = np.array(flux, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    bad_pixel_mask = np.isnan(flux) | (flux < 0) | (flux > threshold) | (variance < 0)
    if skylines is not None and wavelengths is not None:
        wavelengths = np.array(wavelengths)
        delta_wavelength = 5  
        for skyline in skylines:
            indices = np.where(np.abs(wavelengths - skyline) <= delta_wavelength)[0]
            bad_pixel_mask[indices] = True
    bad_indices = np.where(bad_pixel_mask)[0]
    for i in bad_indices:
        left = max(0, i - 4)
        right = min(len(flux), i + 5)
        surrounding_flux = flux[left:i].tolist() + flux[i+1:right].tolist()
        surrounding_flux = [value for value in surrounding_flux if not np.isnan(value)]
        flux[i] = np.mean(surrounding_flux) if surrounding_flux else 0
    for i in range(1, len(flux)-1):
        if np.isnan(flux[i-1]) or np.isnan(flux[i+1]):
            continue
        local_mean = np.mean([flux[i-1], flux[i+1]])
        local_std = np.std([flux[i-1], flux[i+1]])
        if local_std == 0:
            continue
        if (flux[i] - local_mean) > 30 * local_std:
            window_start = max(0, i - 4)
            window_end = min(len(flux), i + 5)
            large_window_start = max(0, i - 9)
            large_window_end = min(len(flux), i + 10)
            surrounding_flux = np.concatenate((flux[large_window_start:window_start],
                                                 flux[window_end:large_window_end]))
            surrounding_flux = surrounding_flux[~np.isnan(surrounding_flux)]
            mean_intensity = np.mean(surrounding_flux) if surrounding_flux.size > 0 else 0
            flux[window_start:window_end] = mean_intensity
    return flux

def subtract_continuum(spectrum, degree=6, sigma_threshold=3, max_iterations=15):
    spectrum = np.array(spectrum)
    x = np.arange(len(spectrum))
    mask = np.ones(len(spectrum), dtype=bool)
    prev_mask = np.zeros(len(spectrum), dtype=bool)
    for iteration in range(max_iterations):
        coeffs = np.polyfit(x[mask], spectrum[mask], degree)
        poly = np.polyval(coeffs, x)
        residuals = spectrum - poly
        std_dev = np.std(residuals[mask])
        prev_mask = mask.copy()
        mask = np.abs(residuals) <= sigma_threshold * std_dev
        if np.array_equal(mask, prev_mask):
            break
    continuum_subtracted = spectrum - poly
    return continuum_subtracted, spectrum - continuum_subtracted

def clip_spectrum_features(spectrum, sigma_threshold=20):
    spectrum = np.array(spectrum)
    mean_flux = np.mean(spectrum)
    std_flux = np.std(spectrum)
    upper_threshold = mean_flux + sigma_threshold * std_flux
    lower_threshold = mean_flux - sigma_threshold * std_flux
    return np.clip(spectrum, lower_threshold, upper_threshold)

# -------------------------------
# Main validation routine (vectorized, GPU friendly)
def batch_validation(batch_size=1000):
    print(f'Using batch size: {batch_size}')
    h5_out_filename = str(validation_config['output_validation_file'])
    
    # Open the input HDF5 file once to get shapes and number of samples
    with h5py.File(validation_config['input_data_file'], 'r') as f:
        total_samples = f['snr'].shape[0]
        print(f"Total samples in data: {total_samples}")
    
    # Load the model once on GPU
    model_name = validation_config['model_name']
    model_path = f"{model_name}.keras"
    if not (os.path.isfile(model_path) and os.access(model_path, os.R_OK)):
        raise FileNotFoundError(f"Model file not found or not readable: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Process batches
    for start in range(0, total_samples, batch_size):
        t_batch_start = time.time()
        print(f'\nProcessing batch starting at index {start}')
        sys.stdout.flush()
        with h5py.File(validation_config['input_data_file'], 'r') as f:
            snr_batch = f['snr'][start:start+batch_size]
            z_true_batch = f['z_true'][start:start+batch_size]
            flux_batch = f['preprocessed_flux'][start:start+batch_size]
            continuum_batch = f['continuum'][start:start+batch_size]
            filenames_batch = f['filename'][start:start+batch_size]
            # Decode filenames if stored as bytes
            filenames_batch = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in filenames_batch]
        
        # Filter out samples with redshift out of range (vectorized filtering)
        valid_mask = [i for i, n in enumerate(z_true_batch) if n < max_z and n > 0.003]#(z_true_batch <= max_z) & (z_true_batch >= 0.003)
        if not np.any(valid_mask):
            print("No valid samples in this batch; skipping.")
            continue
        
        flux_valid = flux_batch[valid_mask]
        z_true_valid = z_true_batch[valid_mask]
        snr_valid = snr_batch[valid_mask]
        continuum_valid = continuum_batch[valid_mask]
        filenames_valid = np.array(filenames_batch)[valid_mask]
        
        # Normalize each flux vector (vectorized along axis 1)
        norms = np.linalg.norm(flux_valid, axis=1, keepdims=True)
        flux_norm = flux_valid / norms
        
        # Run model prediction on the entire batch
        # (Assuming the model expects two identical inputs)
        decoded = model.predict(flux_norm, batch_size=flux_norm.shape[0], verbose=0)
        
        # Compute Gaussian templates for the batch (using TF functions on GPU)
        gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])
        gaussians_norm = tf.norm(gaussians_batch_full, axis=1, keepdims=True)
        gaussians_batch_full = gaussians_batch_full / gaussians_norm
        
        # Compute best shifts based on euclidean distance and cosine similarity;
        # Here we pass the normalized flux (converted to tensor) as the "tiny" array.
        gaussians_tensor = tf.convert_to_tensor(gaussians_batch_full)
        flux_tensor = tf.convert_to_tensor(flux_norm, dtype=tf.float32)
        best_shifts, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_tensor, flux_tensor, radius=1.0, alpha=0.9)
        best_shifts = best_shifts.numpy()
        
        # Convert best shifts into redshift predictions.
        # (For batch sizes that are not huge, a Python loop is acceptable.)
        z_pred_valid = np.array([shift_to_redshift(s, wavelength_template) for s in best_shifts])
        
        # For additional outputs, compute widths and select the corresponding template segment.
        widths_valid = np.array([width_to_velocity(d[-1]) for d in decoded])
        # Optionally, extract the full predicted template per sample.
        templates_valid = gaussians_batch_full.numpy()
        
        # Restore the original flux scale (multiply by saved norm) for output
        flux_reconstructed = flux_norm * norms
        
        
        if start == 0:
            # Remove existing file if it exists
            if os.path.exists(h5_out_filename):
                os.remove(h5_out_filename)


            # Create and save the initial data
            print('Creating new file')
            with h5py.File(h5_out_filename, 'w') as f:
                sys.stdout.flush()
                # Save 'flux_data' dataset
                if len(flux_reconstructed.shape) > 1:
                    maxshape_flux = (None, flux_reconstructed.shape[1])
                else:
                    flux_reconstructed = flux_all.reshape(-1, 1)  # Ensure 2D
                    maxshape_flux = (None, 1)
                dset_flux = f.create_dataset(
                    "flux_data",
                    data=flux_reconstructed,
                    maxshape=maxshape_flux,
                    chunks=(10000, flux_reconstructed.shape[1])
                )

                print('Saving flux data')
                # Save 'continuum_all' dataset
                if len(continuum_valid.shape) > 1:
                    maxshape_continuum = (None, continuum_valid.shape[1])
                else:
                    continuum_valid = continuum_valid.reshape(-1, 1)
                    maxshape_continuum = (None, 1)
                dset_continuum = f.create_dataset(
                    "continuum_all",
                    data=continuum_valid,
                    maxshape=maxshape_continuum,
                    chunks=(10000, continuum_valid.shape[1])
                )
                
                
                if len(hybrid_loss.shape) > 1:
                    maxshape_hybrid_loss = (None, hybrid_loss.shape[1])
                else:
                    hybrid_loss = hybrid_loss.reshape(-1, 1)
                    maxshape_hybrid_loss = (None, 1)
                dset_hybrid_loss = f.create_dataset(
                    "hybrid_loss",
                    data=hybrid_loss,
                    maxshape=maxshape_hybrid_loss,
                    chunks=(10000, hybrid_loss.shape[1])
                )

                print('Saving continuum data')
                # Save 'all_template_data' dataset
                if len(templates_valid.shape) > 1:
                    maxshape_template = (None, templates_valid.shape[1])
                else:
                    templates_valid = templates_valid.reshape(-1, 1)
                    maxshape_template = (None, 1)
                dset_template = f.create_dataset(
                    "all_template_data",
                    data=templates_valid,
                    maxshape=maxshape_template,
                    chunks=(10000, templates_valid.shape[1])
                )

                print('Saving template data')
                # Save 'decoded_all' dataset
                if len(decoded.shape) > 1:
                    maxshape_decoded = (None, decoded.shape[1])
                else:
                    decoded = decoded.reshape(-1, 1)
                    maxshape_decoded = (None, 1)
                dset_decoded = f.create_dataset(
                    "decoded_all",
                    data=decoded,
                    maxshape=maxshape_decoded,
                    chunks=(10000, decoded.shape[1])
                )

                print('Saving decoded data')
                # Save 'snr_all' dataset
                snr_valid = snr_valid.reshape(-1, 1)
                dset_snr = f.create_dataset(
                    "snr_all",
                    data=snr_valid,
                    maxshape=(None, 1),
                    chunks=(10000, 1)
                )

                print('Saving snr data')
                # Save 'filenames_all' dataset
                # Convert filenames_all to a list of Python strings
                filenames_valid = [str(filename) for filename in filenames_valid]
                dt = h5py.string_dtype(encoding='utf-8')
                filenames_valid = np.array(filenames_valid, dtype=dt)
                filenames_valid = filenames_valid.reshape(-1, 1)
                dset_filenames = f.create_dataset(
                    "filenames_all",
                    data=filenames_valid,
                    dtype=dt,
                    maxshape=(None, 1),
                    chunks=(10000, 1)
                )

                print('Saving filenames data')
                # Save 'z_true_all' dataset
                z_true_valid = z_true_valid.reshape(-1, 1)
                dset_z_true = f.create_dataset(
                    "z_true_all",
                    data=z_true_valid,
                    maxshape=(None, 1),
                    chunks=(10000, 1)
                )

                print('Saving z_true data')
                # Save 'z_pred_all' dataset
                z_pred_valid = z_pred_valid.reshape(-1, 1)
                dset_z_pred = f.create_dataset(
                    "z_pred_all",
                    data=z_pred_valid,
                    maxshape=(None, 1),
                    chunks=(10000, 1)
                )
                print('Saving z_pred data')

        else:
            # Append data to existing datasets
            with h5py.File(h5_out_filename, 'a') as f:
                # Append to 'flux_data' dataset
                dset_flux = f['flux_data']
                new_rows = flux_reconstructed.shape[0]
                dset_flux.resize(dset_flux.shape[0] + new_rows, axis=0)
                dset_flux[-new_rows:] = flux_reconstructed

                # Append to 'continuum_all' dataset
                dset_continuum = f['continuum_all']
                new_rows = continuum_valid.shape[0]
                dset_continuum.resize(dset_continuum.shape[0] + new_rows, axis=0)
                dset_continuum[-new_rows:] = continuum_valid

                # Append to 'all_template_data' dataset
                dset_template = f['all_template_data']
                new_rows = templates_valid.shape[0]
                dset_template.resize(dset_template.shape[0] + new_rows, axis=0)
                dset_template[-new_rows:] = templates_valid


                dset_hybrid_loss = f['hybrid_loss']
                new_rows = hybrid_loss.shape[0]
                dset_hybrid_loss.resize(dset_hybrid_loss.shape[0] + new_rows, axis=0)
                dset_hybrid_loss[-new_rows:] = hybrid_loss


                # Append to 'decoded_all' dataset
                dset_decoded = f['decoded_all']
                new_rows = decoded.shape[0]
                dset_decoded.resize(dset_decoded.shape[0] + new_rows, axis=0)
                dset_decoded[-new_rows:] = decoded

                # Append to 'snr_all' dataset
                dset_snr = f['snr_all']
                snr_valid = snr_valid.reshape(-1, 1)
                new_rows = snr_valid.shape[0]
                dset_snr.resize(dset_snr.shape[0] + new_rows, axis=0)
                dset_snr[-new_rows:] = snr_valid

                # Append to 'filenames_all' dataset
                dset_filenames = f['filenames_all']
                # Convert filenames_all to a list of Python strings
                filenames_valid = [str(filename) for filename in filenames_valid]
                filenames_valid = np.array(filenames_valid, dtype=dt)
                filenames_valid = filenames_valid.reshape(-1, 1)
                new_rows = filenames_valid.shape[0]
                dset_filenames.resize(dset_filenames.shape[0] + new_rows, axis=0)
                dset_filenames[-new_rows:] = filenames_valid

                # Append to 'z_true_all' dataset
                dset_z_true = f['z_true_all']
                z_true_valid = z_true_valid.reshape(-1, 1)
                new_rows = z_true_valid.shape[0]
                dset_z_true.resize(dset_z_true.shape[0] + new_rows, axis=0)
                dset_z_true[-new_rows:] = z_true_valid

                # Append to 'z_pred_all' dataset
                dset_z_pred = f['z_pred_all']
                z_pred_valid = z_pred_valid.reshape(-1, 1)
                new_rows = z_pred_valid.shape[0]
                dset_z_pred.resize(dset_z_pred.shape[0] + new_rows, axis=0)
                dset_z_pred[-new_rows:] = z_pred_valid



        
        print(f"Batch processed in {time.time() - t_batch_start:.2f} seconds.")
        sys.stdout.flush()
    

# -------------------------------
if __name__ == '__main__':
    # When using GPU inference, a vectorized approach is preferable to multiprocessing.
    batch_size = validation_config.get('batch_size', 16)
    batch_validation(batch_size=16)
