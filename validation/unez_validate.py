import astropy.io.fits as pyfits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input, Embedding, GlobalAveragePooling1D
import multiprocessing
import time
start_time = time.time()
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt, boxcar, convolve
from scipy.ndimage.filters import uniform_filter1d, median_filter
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, Zeros
import os
import math
import astropy.io.fits as pyfits
import sys
import json
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = 'tmp/matplotlib'
os.environ['TMPDIR'] = 'tmp'

import json
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = 'tmp/matplotlib'
os.environ['TMPDIR'] = 'tmp'

# Ensure the directories exist
os.makedirs('tmp/matplotlib', exist_ok=True)
os.makedirs('tmp', exist_ok=True)
import datetime
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
import astropy.io.fits as pyfits
import pickle
import sys
from itertools import chain
from multiprocessing import Pool
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input, Embedding, GlobalAveragePooling1D
import multiprocessing
import time
start_time = time.time()
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, Zeros
import psutil
#from memory_profiler import profile
import gc
import shutil
import math
import json
import h5py
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



global len_data
global max_z
global wavelength_template
global means
global columns
global validation_config


# Load configuration file
config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Load physics information and training config
physics_info = config['physicsinfo']
validation_config = config['validationinfo']

#maximum expected redshift
max_z = physics_info['max_z']

#number of wavelength bins
len_data = physics_info['len_data']

#maximum observed wavelength
x_max = np.log10(physics_info['x_max'])

#resolution of the wavelength bins (in log10 angstroms)
resolution = physics_info['resolution'] 

#number of pixels in the wavelength template
num_pixels = len_data + math.ceil((np.log10(max_z+1))/resolution)

#generate the model's wavelength template
wavelength_template = np.arange(x_max - resolution*num_pixels, x_max, resolution)

#expected emission/absorption lines
means = np.log10(physics_info['means'])

def fix_names_list(names):
    fixed_names = []
    for name in names:
        if isinstance(name, list):  # Check if the element is a list
            # Convert the list to a string with parentheses included
            fixed_names.append(f"[{', '.join(name)}]")
        else:
            fixed_names.append(name)  # Leave non-list elements unchanged
    return fixed_names

columns = fix_names_list(physics_info['columns']) #HB is idx 24  Ha is 32    






logdir = "tf_logs" 

def inverted_relu(x):
    return -tf.nn.relu(x)  # Negate the output of the standard ReLU


class CustomMatrixMultiplication(Layer):
    def call(self, inputs, **kwargs):
        matrix1, matrix2 = inputs
        # Transpose the matrices to make the inner dimensions match for multiplication
        matrix1_transposed = tf.transpose(matrix1, perm=[0, 2, 1])
        matrix2_transposed = tf.transpose(matrix2, perm=[0, 2, 1])
        # Perform matrix multiplication
        result = tf.matmul(matrix1_transposed, matrix2_transposed)
        # Sum over the last axis (axis 2 here)
        result_sum = tf.reduce_sum(result, axis=-1)
        return result_sum



class ScaledSigmoid(Layer):
    def __init__(self, min_val, max_val, steepness=0.1, **kwargs):
        super(ScaledSigmoid, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.steepness = steepness  # Introduce steepness parameter

    def call(self, inputs, **kwargs):
        # Scale inputs by steepness factor before applying the sigmoid
        sigmoid = tf.nn.sigmoid(inputs * self.steepness)
        # Scale the output of the sigmoid from min_val to max_val
        return self.min_val + (self.max_val - self.min_val) * sigmoid

    def get_config(self):
        config = super(ScaledSigmoid, self).get_config()
        config.update({
            'min_val': self.min_val,
            'max_val': self.max_val,
            'steepness': self.steepness  # Make sure to include steepness in the config
        })
        return config
    



def redshift_to_shift(z, wavelength_template):
    x = wavelength_template  # Assume x is in log10(wavelength)
    obs_wavelength_log = x[-1]
    
    # Calculate delta_log from the redshift
    delta_log = np.log10(1 + z)
    
    # Compute the expected emission wavelength in log scale
    em_wavelength_log = obs_wavelength_log - delta_log
    
    # Find the index in the wavelength template closest to em_wavelength_log
    shift = np.argmin(np.abs(x - em_wavelength_log))
    
    return shift - len_data + 1

def shift_to_redshift(shift, wavelength_template):
    x = wavelength_template  # Assume x is in log10(wavelength)
    obs_wavelength_log = x[-1]
    em_wavelength_log = x[shift + len_data - 1]
    
    # Calculate delta_log
    delta_log = obs_wavelength_log - em_wavelength_log
    
    # Convert delta_log back to redshift
    z = 10 ** delta_log - 1
    
    return z

@tf.function
def compute_batch_gaussians_tf(template, batch_amplitudes, batch_std_devs):
    template = tf.cast(10**template, dtype = tf.float32)   # Units = log10(Anstroms)
    batch_amplitudes = tf.cast(batch_amplitudes, dtype = tf.float32)    #Units = arbitrary flux units
    batch_std_devs = tf.cast(batch_std_devs+5e-7, dtype = tf.float32)    #Units = Anstroms (assuming center at 6500 angstroms)
    # Constants for the means
    means_cbg = tf.constant(10**means, dtype=tf.float32)  #Units = log10(Anstroms)

    # Ensure batch_std_devs is a 1D array
    if len(batch_std_devs.shape) != 1:
        raise ValueError("batch_std_devs must be a 1D array")

    # Expand batch_std_devs to match the dimensions needed for broadcasting
    std_dev_expanded = tf.reshape(batch_std_devs, (-1, 1, 1))  # [B, 1, 1]

    # Compute Gaussian distributions
    expanded_template = tf.expand_dims(template, 1)  # [N, 1]
    expanded_means = tf.expand_dims(means_cbg, 0)  # [1, M]

    # Apply broadcasting to calculate the Gaussians
    gaussians = (1/(std_dev_expanded*tf.math.sqrt(2*math.pi)))*tf.exp(-0.5 * tf.square((expanded_template - expanded_means) / std_dev_expanded))  # [B, N, M]

    # Transpose and expand gaussians for correct broadcasting
    gaussians = tf.transpose(gaussians, perm=[0, 2, 1])  # [B, M, N]

    # Expand batch amplitudes
    batch_amplitudes_expanded = tf.expand_dims(batch_amplitudes, 2)  # [B, M, 1]

    # Multiply Gaussians by batch amplitudes
    gaussians_scaled = gaussians * batch_amplitudes_expanded  # [B, M, N]

    # Sum along the means axis
    summed_gaussians = tf.reduce_sum(gaussians_scaled, axis=1)  # [B, N]

    return summed_gaussians


@tf.function
def slice_2d_tensor_by_1d_indices(data_2d, indices_1d, data_length = len_data):
    # Calculate continuous indices within allowed bounds
    idx_min = indices_1d 
    idx_max = idx_min + data_length

    max_len = tf.cast(tf.shape(data_2d)[1], tf.float32)
    idx_max = tf.minimum(idx_max, max_len)

    # Create a meshgrid for the batch and indices
    idx_range = tf.linspace(0.0, 1.0, len_data)  # Create len_data points between 0 and 1
    idx_range = tf.expand_dims(idx_range, 0)  # Shape: [1, len_data]

    # Interpolate between idx_min and idx_max
    idxs = idx_min[:, None] + idx_range * (idx_max - idx_min)[:, None]
    idxs = tf.clip_by_value(idxs, 0.0, max_len - 1.0)  # Ensure indices are within valid range

    # Perform bilinear interpolation
    idx_floor = tf.floor(idxs)
    idx_ceil = idx_floor + 1
    idx_ceil = tf.minimum(idx_ceil, max_len - 1.0)  # Ensure idx_ceil does not exceed data length

    idx_floor = tf.cast(idx_floor, tf.int32)
    idx_ceil = tf.cast(idx_ceil, tf.int32)

    # Get values at idx_floor and idx_ceil
    def gather_vals(data, indices):
        batch_indices = tf.tile(tf.range(tf.shape(data)[0])[:, None], [1, tf.shape(indices)[1]])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        return tf.gather_nd(data, gather_indices)

    values_floor = gather_vals(data_2d, idx_floor)
    values_ceil = gather_vals(data_2d, idx_ceil)

    # Calculate the weights for interpolation
    weights = idxs - tf.cast(idx_floor, tf.float32)

    # Interpolate between floor and ceil values
    result_tensor = values_floor * (1.0 - weights) + values_ceil * weights

    return result_tensor



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


def exponential_decay_radius(epoch, initial_radius, min_radius, total_epochs):
    """
    Calculate the radius decay using an exponential function.

    :param epoch: Current epoch (zero-indexed).
    :param initial_radius: The starting value of the radius.
    :param min_radius: The minimum value the radius can reach.
    :param total_epochs: The total number of epochs over which the decay happens.
    :return: The adjusted radius for the given epoch.
    """
    if epoch >= total_epochs:
        return min_radius
    decay_rate = (initial_radius / min_radius) ** (1 / total_epochs)
    return initial_radius * (decay_rate ** epoch)




def width_to_velocity(width):
    center = 6500
    velocity = ((width)/center)*300000

    return velocity

#dz/(1+z) * c

def init_worker():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This makes GPUs invisible to the subprocesses


def preprocess_flux(flux, variance, threshold, wavelengths=None, skylines=None):
    """
    Preprocess flux data to remove bad pixels, cosmic rays, and mask known skylines.
    
    Parameters:
    - flux: List or numpy array of flux values.
    - variance: List or numpy array of variance values.
    - threshold: Configurable threshold for identifying bad pixels.
    - wavelengths: (Optional) List or numpy array of wavelengths corresponding to flux values.
    - skylines: (Optional) List of wavelengths to mask that are specifically known skylines.

    Returns:
    - A numpy array of cleaned flux data.
    """
    flux = np.array(flux, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    
    # Identify bad pixels based on criteria: NaN, negative, exceeds threshold, or negative variance
    bad_pixel_mask = np.isnan(flux) | (flux < 0) | (flux > threshold) | (variance < 0)
    
    # Mask skylines if provided
    if skylines is not None and wavelengths is not None:
        wavelengths = np.array(wavelengths)
        # Define a tolerance for matching skylines (adjust as necessary)
        delta_wavelength = 5  # Example tolerance value
        # Create a boolean mask for skylines
        for skyline in skylines:
            indices = np.where(np.abs(wavelengths - skyline) <= delta_wavelength)[0]
            bad_pixel_mask[indices] = True

    # Replace bad pixels with the mean of four pixels on either side
    bad_indices = np.where(bad_pixel_mask)[0]
    for i in bad_indices:
        left = max(0, i - 4)
        right = min(len(flux), i + 5)  # +5 because slicing is exclusive of the end index
        # Exclude the bad pixel itself and any other bad pixels in the window
        surrounding_flux = flux[left:i].tolist() + flux[i+1:right].tolist()
        surrounding_flux = [value for value in surrounding_flux if not np.isnan(value)]
        if surrounding_flux:
            flux[i] = np.mean(surrounding_flux)
        else:
            flux[i] = 0  # Assign zero if no good surrounding pixels are found
    
    # Identify cosmic rays
    for i in range(1, len(flux) - 1):
        # Skip if neighboring pixels are NaN
        if np.isnan(flux[i-1]) or np.isnan(flux[i+1]):
            continue
        local_mean = np.mean([flux[i-1], flux[i+1]])
        local_std = np.std([flux[i-1], flux[i+1]])
        if local_std == 0:
            continue  # Avoid division by zero
        if (flux[i] - local_mean) > 30 * local_std:
            # Replace 9-pixel window centered on the cosmic ray pixel with mean intensity from 19-pixel window
            window_start = max(0, i - 4)
            window_end = min(len(flux), i + 5)
            large_window_start = max(0, i - 9)
            large_window_end = min(len(flux), i + 10)

            # Exclude pixels in the 9-pixel window and any NaNs
            surrounding_flux = np.concatenate((
                flux[large_window_start:window_start],
                flux[window_end:large_window_end]
            ))
            surrounding_flux = surrounding_flux[~np.isnan(surrounding_flux)]
            if surrounding_flux.size > 0:
                mean_intensity = np.mean(surrounding_flux)
            else:
                mean_intensity = 0  # Assign zero if no good surrounding pixels are found
            
            # Replace the 9-pixel window
            flux[window_start:window_end] = mean_intensity

    return flux


def subtract_continuum(spectrum, degree=6, sigma_threshold=3, max_iterations=15):
    """
    Subtract the continuum from a spectrum using rejected polynomial subtraction.
    
    Parameters:
    - spectrum: List or numpy array of flux values representing the spectrum.
    - degree: The degree of the polynomial to fit (default is 6).
    - sigma_threshold: The standard deviation threshold for rejecting points (default is 3.5).
    - max_iterations: Maximum number of iterations for the fitting process (default is 15).
    
    Returns:
    - The spectrum with the continuum subtracted.
    """
    # Ensure the input is a numpy array
    spectrum = np.array(spectrum)
    x = np.arange(len(spectrum))

    # Initialize variables for the fitting process
    mask = np.ones(len(spectrum), dtype=bool)  # Start with all points included
    prev_mask = np.zeros(len(spectrum), dtype=bool)

    for iteration in range(max_iterations):
        # Fit a polynomial to the currently unmasked points
        coeffs = np.polyfit(x[mask], spectrum[mask], degree)
        poly = np.polyval(coeffs, x)

        # Calculate the residuals and standard deviation of the residuals
        residuals = spectrum - poly
        std_dev = np.std(residuals[mask])

        # Update the mask by rejecting points more than sigma_threshold * std_dev away from the polynomial
        prev_mask = mask.copy()
        mask = np.abs(residuals) <= sigma_threshold * std_dev

        # Check for convergence (no more points being rejected)
        if np.array_equal(mask, prev_mask):
            break

    # Subtract the fitted polynomial from the original spectrum to remove the continuum
    continuum_subtracted_spectrum = spectrum - poly

    return continuum_subtracted_spectrum, spectrum - continuum_subtracted_spectrum



def clip_spectrum_features(spectrum, sigma_threshold=20):
    """
    Clip all features in a spectrum at a specified number of standard deviations from the mean.
    
    Parameters:
    - spectrum: List or numpy array of flux values representing the spectrum.
    - sigma_threshold: The standard deviation threshold for clipping (default is 30).
    
    Returns:
    - The spectrum with extreme features clipped.
    """
    # Convert the input to a numpy array
    spectrum = np.array(spectrum)

    # Calculate the mean and standard deviation of the spectrum
    mean_flux = np.mean(spectrum)
    std_flux = np.std(spectrum)

    # Define the upper and lower clipping thresholds
    upper_threshold = mean_flux + sigma_threshold * std_flux
    lower_threshold = mean_flux - sigma_threshold * std_flux

    # Clip the spectrum values to be within the threshold limits
    clipped_spectrum = np.clip(spectrum, lower_threshold, upper_threshold)
    
    return clipped_spectrum

class GlobalSumPooling1D(Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)



z_true = []
z_pred = []

custom_objects = {
    'ScaledSigmoid': ScaledSigmoid,
    'GlobalSumPooling1D': GlobalSumPooling1D,
    'inverted_relu': inverted_relu
}
def process_file(flux, redshift, SNR, filename, continuum):
    try:
        sys.stdout.flush()
        
        if redshift > max_z or redshift < 0.003:
            return None, None, None, None, None, None, None, None, None, None
            
        norm = np.linalg.norm(flux)

        # Normalize the array
        if norm != 0:
            flux = flux / norm
        else:
            flux = flux  # Handle the case where the norm is zero

        true_shift = redshift_to_shift(redshift, wavelength_template)

        data = flux

        sys.stdout.flush()
        # Predict using the model
        decoded = model.predict(np.array([data, data]), verbose=0)
        
        sys.stdout.flush()

        gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])
        sys.stdout.flush()
        gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
        gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm
        best_shifts, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, [data, data], radius = 1, alpha = 0.9)
        sys.stdout.flush()

        z_true.append(redshift)
        z_pred.append(shift_to_redshift(best_shifts[0], wavelength_template))
        sys.stdout.flush()

        flux_predicted_trf = gaussians_batch_full[0][redshift_to_shift(0, wavelength_template):redshift_to_shift(0, wavelength_template)+len_data]

        flux_predicted = gaussians_batch_full[0][best_shifts[0]:best_shifts[0]+len_data]
        sys.stdout.flush()

        object_class = 'QSO'
        sys.stdout.flush()

        return redshift[0], shift_to_redshift(best_shifts[0], wavelength_template), flux*norm, width_to_velocity(decoded[0][-1]), decoded[0], SNR, filename, continuum, object_class, gaussians_batch_full[0]
    
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        return None, None, None, None, None, None, None, None, None, None
        



def init_worker():
    global model
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Prevent TensorFlow from using GPU in subprocesses
    model_name = validation_config['model_name']
    model_path = f"{model_name}.keras"

    if os.path.isfile(model_path) and os.access(model_path, os.R_OK):
        # Load the model and provide the custom objects
        model = load_model(model_path, custom_objects=custom_objects)
    else:
        raise FileNotFoundError(f"File not found or not readable: {model_path}")



def batch_validation(batch_size=1000):
    print(f'batch size: {batch_size}')
    sys.stdout.flush()
    h5_filename = str(validation_config['output_validation_file'])  # Name of the single HDF5 file
    
    with h5py.File(validation_config['input_data_file'], 'r') as f:
        # List all datasets in the file
        print("Datasets in the file:")
        
        for key in f.keys():
            print(key)

        sys.stdout.flush()

        print('reading snr')
        sys.stdout.flush()
        snr_all = f['snr']
        print(f'snr_all shape: {snr_all.shape}')

    
    for batch in np.arange(0, len(snr_all), batch_size):
        start_time = time.time()
        
        with h5py.File(validation_config['input_data_file'], 'r') as f:
            # List all datasets in the file
            

            sys.stdout.flush()

            sys.stdout.flush()
            snr_all = f['snr'][batch:batch+batch_size]



            z_true_all = f['z_true'][batch:batch+batch_size]



            sys.stdout.flush()
            flux_all = f['preprocessed_flux'][batch:batch+batch_size] 

            continuum_all = f['continuum'][batch:batch+batch_size]
            sys.stdout.flush()
            

            # Read the 'filenames_all' dataset and decode the strings
            filenames_all = f['filename'][batch:batch+batch_size]
            # Since 'filenames_all' was saved as an array of arrays, flatten and decode
            filenames_all = [name.decode('utf-8') for name in filenames_all]


        
        # Create a pool of processes
        with multiprocessing.Pool(initializer=init_worker) as pool:
            results = pool.starmap(process_file, [(flux, redshift, SNR, filename, continuum) for flux, redshift, SNR, filename, continuum in zip(flux_all, z_true_all, snr_all, filenames_all, continuum_all)])  
        # Aggregate the results
        z_true_all = np.array([row[0] for row in results if row[0] is not None])        # true redshift #
        z_pred_all = np.array([row[1] for row in results if row[1] is not None])        
        flux_all = np.array([row[2] for row in results if row[2] is not None])          # interpolated flux not normed#
        widths_all = np.array([row[3] for row in results if row[3] is not None])        
        decoded_all = np.array([row[4] for row in results if row[4] is not None])       ##  model output
        snr_all = np.array([row[5] for row in results if row[5] is not None])           # snr#
        filenames_all = [row[6] for row in results if row[6] is not None]  
        continuum_all = np.array([row[7] for row in results if row[7] is not None])     # continuum
        class_all = np.array([row[8] for row in results if row[8] is not None])         # class
        all_template_data = np.array([row[9] for row in results if row[9] is not None]) ##  full templates
        print(f'Batch {batch} completed')  
        sys.stdout.flush() 
        print('SAVING FILES')
        sys.stdout.flush()
        end_time = time.time()
        print(end_time - start_time)
        print(f'flux shape: {flux_all.shape}')
        
        if batch == 0:
            # Remove existing file if it exists
            if os.path.exists(h5_filename):
                os.remove(h5_filename)

            # Create and save the initial data
            print('Creating new file')
            with h5py.File(h5_filename, 'w') as f:
                print(h5_filename)
                sys.stdout.flush()
                # Save 'flux_data' dataset
                if len(flux_all.shape) > 1:
                    maxshape_flux = (None, flux_all.shape[1])
                else:
                    flux_all = flux_all.reshape(-1, 1)  # Ensure 2D
                    maxshape_flux = (None, 1)
                dset_flux = f.create_dataset(
                    "flux_data",
                    data=flux_all,
                    maxshape=maxshape_flux,
                    chunks=True
                )

                print('Saving flux data')
                # Save 'continuum_all' dataset
                if len(continuum_all.shape) > 1:
                    maxshape_continuum = (None, continuum_all.shape[1])
                else:
                    continuum_all = continuum_all.reshape(-1, 1)
                    maxshape_continuum = (None, 1)
                dset_continuum = f.create_dataset(
                    "continuum_all",
                    data=continuum_all,
                    maxshape=maxshape_continuum,
                    chunks=True
                )

                print('Saving continuum data')
                # Save 'all_template_data' dataset
                if len(all_template_data.shape) > 1:
                    maxshape_template = (None, all_template_data.shape[1])
                else:
                    all_template_data = all_template_data.reshape(-1, 1)
                    maxshape_template = (None, 1)
                dset_template = f.create_dataset(
                    "all_template_data",
                    data=all_template_data,
                    maxshape=maxshape_template,
                    chunks=True
                )

                print('Saving template data')
                # Save 'decoded_all' dataset
                if len(decoded_all.shape) > 1:
                    maxshape_decoded = (None, decoded_all.shape[1])
                else:
                    decoded_all = decoded_all.reshape(-1, 1)
                    maxshape_decoded = (None, 1)
                dset_decoded = f.create_dataset(
                    "decoded_all",
                    data=decoded_all,
                    maxshape=maxshape_decoded,
                    chunks=True
                )

                print('Saving decoded data')
                # Save 'snr_all' dataset
                snr_all = snr_all.reshape(-1, 1)
                dset_snr = f.create_dataset(
                    "snr_all",
                    data=snr_all,
                    maxshape=(None, 1),
                    chunks=True
                )

                print('Saving snr data')
                # Save 'filenames_all' dataset
                # Convert filenames_all to a list of Python strings
                filenames_all = [str(filename) for filename in filenames_all]
                dt = h5py.string_dtype(encoding='utf-8')
                filenames_all = np.array(filenames_all, dtype=dt)
                filenames_all = filenames_all.reshape(-1, 1)
                dset_filenames = f.create_dataset(
                    "filenames_all",
                    data=filenames_all,
                    dtype=dt,
                    maxshape=(None, 1),
                    chunks=True
                )

                print('Saving filenames data')
                # Save 'z_true_all' dataset
                z_true_all = z_true_all.reshape(-1, 1)
                dset_z_true = f.create_dataset(
                    "z_true_all",
                    data=z_true_all,
                    maxshape=(None, 1),
                    chunks=True
                )

                print('Saving z_true data')
                # Save 'z_pred_all' dataset
                z_pred_all = z_pred_all.reshape(-1, 1)
                dset_z_pred = f.create_dataset(
                    "z_pred_all",
                    data=z_pred_all,
                    maxshape=(None, 1),
                    chunks=True
                )
                print('Saving z_pred data')

                # You can similarly save other datasets as needed

        else:
            # Append data to existing datasets
            with h5py.File(h5_filename, 'a') as f:
                # Append to 'flux_data' dataset
                dset_flux = f['flux_data']
                new_rows = flux_all.shape[0]
                dset_flux.resize(dset_flux.shape[0] + new_rows, axis=0)
                dset_flux[-new_rows:] = flux_all

                # Append to 'continuum_all' dataset
                dset_continuum = f['continuum_all']
                new_rows = continuum_all.shape[0]
                dset_continuum.resize(dset_continuum.shape[0] + new_rows, axis=0)
                dset_continuum[-new_rows:] = continuum_all

                # Append to 'all_template_data' dataset
                dset_template = f['all_template_data']
                new_rows = all_template_data.shape[0]
                dset_template.resize(dset_template.shape[0] + new_rows, axis=0)
                dset_template[-new_rows:] = all_template_data

                # Append to 'decoded_all' dataset
                dset_decoded = f['decoded_all']
                new_rows = decoded_all.shape[0]
                dset_decoded.resize(dset_decoded.shape[0] + new_rows, axis=0)
                dset_decoded[-new_rows:] = decoded_all

                # Append to 'snr_all' dataset
                dset_snr = f['snr_all']
                snr_all = snr_all.reshape(-1, 1)
                new_rows = snr_all.shape[0]
                dset_snr.resize(dset_snr.shape[0] + new_rows, axis=0)
                dset_snr[-new_rows:] = snr_all

                # Append to 'filenames_all' dataset
                dset_filenames = f['filenames_all']
                # Convert filenames_all to a list of Python strings
                filenames_all = [str(filename) for filename in filenames_all]
                filenames_all = np.array(filenames_all, dtype=dt)
                filenames_all = filenames_all.reshape(-1, 1)
                new_rows = filenames_all.shape[0]
                dset_filenames.resize(dset_filenames.shape[0] + new_rows, axis=0)
                dset_filenames[-new_rows:] = filenames_all

                # Append to 'z_true_all' dataset
                dset_z_true = f['z_true_all']
                z_true_all = z_true_all.reshape(-1, 1)
                new_rows = z_true_all.shape[0]
                dset_z_true.resize(dset_z_true.shape[0] + new_rows, axis=0)
                dset_z_true[-new_rows:] = z_true_all

                # Append to 'z_pred_all' dataset
                dset_z_pred = f['z_pred_all']
                z_pred_all = z_pred_all.reshape(-1, 1)
                new_rows = z_pred_all.shape[0]
                dset_z_pred.resize(dset_z_pred.shape[0] + new_rows, axis=0)
                dset_z_pred[-new_rows:] = z_pred_all





if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')  # or 'forkserver'

    
    
    print(validation_config['batch_size'])
    batch_validation(batch_size = validation_config['batch_size'])
    
