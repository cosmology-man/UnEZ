import numpy as np               # For numerical operations, especially array manipulation
import multiprocessing           # For parallel processing to speed up computations
import yaml                      # For loading configuration settings from a YAML file
import os                        # For interacting with the operating system (e.g., file paths, environment variables)
import math                      # For mathematical functions (e.g., ceil, log10)
import sys                       # For system-specific parameters and functions (e.g., stdout.flush)
import h5py                      # For reading and writing HDF5 files, a format suitable for large datasets
from multiprocessing import Pool # Specific import for creating a pool of worker processes

# Prevent Python from generating .pyc files, useful in some environments
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


# --- Global Variables ---
# These variables are defined globally for easy access within different functions,
# particularly within the multiprocessing context.

global len_data              # Expected length (number of bins) of the final processed spectrum data
global max_z                 # Maximum expected redshift value for filtering and processing
global wavelength_template   # The target wavelength grid (in log10 angstroms) onto which spectra will be interpolated
global x_max                 # Maximum wavelength value (log10 angstroms) in the target template
global resolution            # Wavelength resolution (step size) of the target template (log10 angstroms)
global preprocessing_config  # Dictionary holding preprocessing configuration parameters

# --- Configuration Loading ---
print("Loading configuration from config.yaml...")
config_file_path = 'config.yaml'
try:
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully.")
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found.")
    sys.exit(1) # Exit if config file is missing
except yaml.YAMLError as e:
    print(f"Error parsing configuration file: {e}")
    sys.exit(1) # Exit if config file is invalid

# Load specific sections from the configuration
physics_info = config.get('physicsinfo', {}) # Use .get for safer access
preprocessing_config = config.get('preprocessinginfo', {})
if not physics_info or not preprocessing_config:
    print("Error: 'physicsinfo' or 'preprocessinginfo' section missing in config.yaml")
    sys.exit(1)

# --- Parameter Initialization from Config ---
# Maximum expected redshift (used for filtering and determining wavelength range)
max_z = physics_info.get('max_z')
if max_z is None:
    print("Error: 'max_z' not found in physicsinfo configuration.")
    sys.exit(1)

# Number of wavelength bins in the final output spectrum
len_data = physics_info.get('len_data')
if len_data is None:
    print("Error: 'len_data' not found in physicsinfo configuration.")
    sys.exit(1)

# Maximum observed wavelength (log10 angstroms) for the template grid
x_max = np.log10(physics_info.get('x_max'))
if physics_info.get('x_max') is None: # Check before log10
     print("Error: 'x_max' not found in physicsinfo configuration.")
     sys.exit(1)

# Resolution (step size in log10 angstroms) of the wavelength template
resolution = physics_info.get('resolution')
if resolution is None:
    print("Error: 'resolution' not found in physicsinfo configuration.")
    sys.exit(1)

print(f"Parameters loaded: max_z={max_z}, len_data={len_data}, x_max={10**x_max:.1f} A (log10={x_max:.4f}), resolution={resolution:.4e}")

# --- Wavelength Template Generation ---
# Calculate the total number of pixels needed for the template, considering the maximum redshift
num_pixels = len_data + math.ceil((np.log10(max_z + 1)) / resolution)
print(f"Calculated number of pixels for full wavelength template: {num_pixels}")

# Generate the model's wavelength template using numpy's arange
wavelength_template = np.arange(x_max - resolution * num_pixels, x_max, resolution)
print(f"Generated wavelength template with {len(wavelength_template)} bins, from log10({wavelength_template[0]:.4f}) to log10({wavelength_template[-1]:.4f}) A.")

# --- Preprocessing Functions ---
def preprocess_flux(flux, variance, threshold, wavelengths=None, skylines=None):
    """
    Preprocess flux data to remove bad pixels, cosmic rays, and mask known skylines.
    """
    flux = np.array(flux, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    bad_pixel_mask = np.isnan(flux) | (flux < 0) | (flux > threshold) | (variance < 0)
    initial_bad_pixels = np.sum(bad_pixel_mask)

    if skylines is not None and wavelengths is not None:
        wavelengths = np.array(wavelengths)
        delta_wavelength = preprocessing_config.get('skyline_mask_width_angstroms', 5)
        for skyline in skylines:
            indices = np.where(np.abs(wavelengths - skyline) <= delta_wavelength)[0]
            bad_pixel_mask[indices] = True
        skylines_masked = np.sum(bad_pixel_mask) - initial_bad_pixels

    bad_indices = np.where(bad_pixel_mask)[0]
    replaced_count = 0
    for i in bad_indices:
        left = max(0, i - 4)
        right = min(len(flux), i + 5)
        window_indices = list(range(left, i)) + list(range(i + 1, right))
        valid_window_indices = [idx for idx in window_indices if not bad_pixel_mask[idx]]
        if valid_window_indices:
            mean_surrounding_flux = np.mean(flux[valid_window_indices])
            flux[i] = mean_surrounding_flux
            replaced_count += 1
        else:
            flux[i] = 0

    cosmic_ray_count = 0
    cr_sigma_threshold = preprocessing_config.get('cosmic_ray_sigma', 30)
    for i in range(1, len(flux) - 1):
        if bad_pixel_mask[i] or np.isnan(flux[i-1]) or np.isnan(flux[i+1]):
            continue
        local_mean = np.mean([flux[i-1], flux[i+1]])
        local_std = np.std([flux[i-1], flux[i+1]])
        if local_std == 0:
            continue
        if (flux[i] - local_mean) > cr_sigma_threshold * local_std:
            cosmic_ray_count += 1
            window_start = max(0, i - 4)
            window_end = min(len(flux), i + 5)
            large_window_start = max(0, i - 9)
            large_window_end = min(len(flux), i + 10)
            surrounding_flux_indices = list(range(large_window_start, window_start)) + list(range(window_end, large_window_end))
            surrounding_flux = flux[surrounding_flux_indices]
            surrounding_flux = surrounding_flux[~np.isnan(surrounding_flux)]
            if surrounding_flux.size > 0:
                mean_intensity = np.mean(surrounding_flux)
            else:
                mean_intensity = 0
            flux[window_start:window_end] = mean_intensity

    return flux


def subtract_continuum(spectrum, degree=6, sigma_threshold=3.5, max_iterations=15):
    """
    Subtract the continuum from a spectrum using iterative polynomial fitting with sigma clipping.
    """
    spectrum = np.array(spectrum)
    x = np.arange(len(spectrum))
    mask = np.ones(len(spectrum), dtype=bool)
    prev_mask = np.zeros(len(spectrum), dtype=bool)

    for iteration in range(max_iterations):
        if np.sum(mask) <= degree:
            break
        try:
            coeffs = np.polyfit(x[mask], spectrum[mask], degree)
        except (np.linalg.LinAlgError, ValueError) as e:
            break
        poly = np.polyval(coeffs, x)
        residuals = spectrum - poly
        std_dev = np.std(residuals[mask])
        if std_dev < 1e-10:
            break
        prev_mask = mask.copy()
        mask = np.abs(residuals) <= sigma_threshold * std_dev
        if np.array_equal(mask, prev_mask):
            break

    continuum_subtracted_spectrum = spectrum - poly
    return continuum_subtracted_spectrum, poly


def clip_spectrum_features(spectrum, sigma_threshold=30):
    """
    Clip extreme flux values (both high and low) in a spectrum based on standard deviations.
    """
    spectrum = np.array(spectrum)
    mean_flux = np.mean(spectrum)
    std_flux = np.std(spectrum)
    if std_flux == 0:
        return spectrum
    upper_threshold = mean_flux + sigma_threshold * std_flux
    lower_threshold = mean_flux - sigma_threshold * std_flux
    clipped_spectrum = np.clip(spectrum, lower_threshold, upper_threshold)
    return clipped_spectrum


def preprocess_pipeline(flux, ivar, loglam, skylines=[5578.5, 5894.6, 6301.7, 7246.0, 5581.5,
                                                      5582, 5580, 5579.5, 5579, 9376, 9325, 9315,
                                                      9478, 9521, 9505, 9569, 9790, 10015, 10125,
                                                      10175, 10191, 10212, 10287, 10297]):
    """
    Applies a sequence of preprocessing steps to a single spectrum.
    """
    variance = np.divide(1.0, ivar, out=np.full_like(ivar, np.inf), where=ivar!=0)
    bad_pixel_threshold = preprocessing_config.get('bad_pixel_threshold', 1000)
    wavelengths_angstrom = 10**loglam
    flux_f = preprocess_flux(flux, variance, bad_pixel_threshold, wavelengths=wavelengths_angstrom, skylines=skylines)

    poly_degree = preprocessing_config.get('continuum_poly_degree', 6)
    sigma_clip = preprocessing_config.get('continuum_sigma_clip', 3.5)
    max_iter = preprocessing_config.get('continuum_max_iter', 15)
    flux_f, continuum = subtract_continuum(flux_f, degree=poly_degree, sigma_threshold=sigma_clip, max_iterations=max_iter)

    clip_sigma = preprocessing_config.get('feature_clip_sigma', 30)
    flux_f = clip_spectrum_features(flux_f, sigma_threshold=clip_sigma)

    edge_clip_pixels = preprocessing_config.get('edge_clip_pixels', 75)
    if edge_clip_pixels > 0 and len(flux_f) > 2 * edge_clip_pixels:
        flux_f[-edge_clip_pixels:] = 0
        flux_f[:edge_clip_pixels] = 0

    target_loglam = wavelength_template[-(len_data):]
    target_loglam_original = wavelength_template[-len_data-1:-1]
    if np.any(np.isnan(flux_f)) or np.any(np.isinf(flux_f)):
        return None
    try:
        flux_interpolated = np.interp(target_loglam_original, loglam, flux_f, left=0, right=0)
        continuum_interpolated = np.interp(target_loglam_original, loglam, continuum, left=0, right=0)

    except ValueError as e:
        return None

    if np.all(flux_interpolated == 0):
        pass
    return flux_interpolated, continuum_interpolated


def process_file(inputs):
    """
    Worker function for multiprocessing. Takes input data for one spectrum,
    applies quality cuts, runs the preprocessing pipeline, and returns results.
    Now includes the specobjid as a string.
    """
    flux, loglam, redshift, SNR, ivar, specobjid, filename, obj_class = inputs
    min_snr_cut = preprocessing_config.get('min_snr', 0)
    max_snr_cut = preprocessing_config.get('max_snr', float('inf'))
    min_z_cut = preprocessing_config.get('min_z', 0.003)
    max_z_cut = physics_info.get('max_z', float('inf'))

    if not (min_snr_cut <= SNR <= max_snr_cut and min_z_cut <= redshift <= max_z_cut):
        return None, None, None, None, None, None, None

    processed_flux, continuum = preprocess_pipeline(flux, ivar, loglam)
    if processed_flux is None:
        return None, None, None, None, None, None, None

    if len(processed_flux) != len_data:
        return None, None, None, None, None, None, None
    sys.stdout.flush()
    
    return redshift, processed_flux, SNR, continuum, str(specobjid), str(filename), str(obj_class)


# --- Main Execution Block ---
if __name__ == '__main__':
    print("\n--- Starting Main Execution ---")

    files_to_process = preprocessing_config.get('num_files', None)
    raw_data_filename = preprocessing_config.get('raw_data_file')
    output_filename = preprocessing_config.get('output_file')

    if not raw_data_filename or not output_filename:
        print("Error: 'raw_data_file' or 'output_file' not specified in preprocessinginfo config.")
        sys.exit(1)

    raw_data_filepath = os.path.join('data', raw_data_filename)
    output_filepath = os.path.join('data', output_filename)

    print(f"Input HDF5 file: {raw_data_filepath}")
    print(f"Output HDF5 file: {output_filepath}")

    # Read batch_size from configuration (if not set, default to process all spectra in one batch)
    batch_size = preprocessing_config.get('batch_size', None)

    # --- Open Input and Output HDF5 Files ---
    try:
        with h5py.File(raw_data_filepath, 'r') as f_in:
            print(f"\nDatasets available in the input file:")
            for key in f_in.keys():
                print(f"- {key}")

            num_available = len(f_in['z_true_all'])
            print(f"Total spectra available in file: {num_available}")

            if files_to_process is None or files_to_process > num_available:
                files_to_load = num_available
                if files_to_process is not None and files_to_process > num_available:
                    print(f"Warning: Requested {files_to_process} spectra, but only {num_available} are available. Processing all available.")
            else:
                files_to_load = files_to_process

            # If batch_size is not set or is larger than files_to_load, process all at once
            if batch_size is None or batch_size > files_to_load:
                batch_size = files_to_load

            print(f"Processing {files_to_load} spectra in batches of {batch_size}.")

            # Open output file in write mode and create extendable datasets
            with h5py.File(output_filepath, 'w') as f_out:
                dset_flux = f_out.create_dataset(
                    "preprocessed_flux",
                    shape=(0, len_data),
                    maxshape=(None, len_data),
                    chunks=(min(100, files_to_load), len_data)
                )
                dset_z = f_out.create_dataset(
                    "z_true",
                    shape=(0, 1),
                    maxshape=(None, 1),
                    chunks=(min(10000, files_to_load), 1)
                )
                dset_snr = f_out.create_dataset(
                    "snr",
                    shape=(0, 1),
                    maxshape=(None, 1),
                    chunks=(min(10000, files_to_load), 1)
                )
                dset_continuum = f_out.create_dataset(
                    "continuum",
                    shape=(0, len_data),
                    maxshape=(None, len_data),
                    chunks=(min(100, files_to_load), len_data)
                )
                # Create a dataset for specobjid as strings
                dset_specobjid = f_out.create_dataset(
                    "specobjid",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(min(10000, files_to_load),)
                )
                # Create a dataset for filename as strings
                dset_filename = f_out.create_dataset(
                    "filename",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(min(10000, files_to_load),)
                )
                dset_class = f_out.create_dataset(
                    "class",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(min(10000, files_to_load),)
                )
                total_valid = 0
                total_skipped = 0

                # Create references to the input datasets
                snr_ds = f_in['snr_all']
                flux_ds = f_in['flux_raw_all']
                loglam_ds = f_in['loglam_all']
                redshift_ds = f_in['z_true_all']
                ivar_ds = f_in['ivar_all']
                specobjid_ds = f_in['specobjid_all']
                filename_ds = f_in['filenames_all']
                class_ds = f_in['class_all']

                # Initialize multiprocessing pool once
                with Pool() as pool:
                    # Process spectra in batches
                    for start in range(0, files_to_load, batch_size):
                        end = min(start + batch_size, files_to_load)
                        print(f"\nProcessing batch: spectra {start} to {end - 1}...")
                        snr_batch = snr_ds[start:end]
                        flux_batch = flux_ds[start:end]
                        loglam_batch = loglam_ds[start:end]
                        redshift_batch = redshift_ds[start:end]
                        ivar_batch = ivar_ds[start:end]
                        specobjid_batch = specobjid_ds[start:end]
                        filename_batch = filename_ds[start:end]
                        class_batch = class_ds[start:end]

                        # Apply initial SNR pre-filter for the batch
                        min_snr_precut = preprocessing_config.get('min_snr', 0)
                        snr_mask = snr_batch >= min_snr_precut
                        loglam_filtered = loglam_batch[snr_mask]
                        flux_filtered = flux_batch[snr_mask]
                        redshift_filtered = redshift_batch[snr_mask]
                        snr_filtered = snr_batch[snr_mask]
                        ivar_filtered = ivar_batch[snr_mask]
                        specobjid_filtered = specobjid_batch[snr_mask]
                        filename_filtered = filename_batch[snr_mask]
                        class_filtered = class_batch[snr_mask]

                        num_in_batch = len(snr_filtered)
                        print(f"Spectra remaining after SNR pre-filter in this batch: {num_in_batch} (out of {end - start})")

                        if num_in_batch == 0:
                            print("No spectra passed the SNR pre-filter in this batch. Skipping batch.")
                            total_skipped += (end - start)
                            continue

                        # Prepare inputs for multiprocessing (including specobjid)
                        inputs_for_pool = zip(flux_filtered, loglam_filtered, redshift_filtered, snr_filtered, ivar_filtered, 
                                              specobjid_filtered, filename_filtered, class_filtered)

                        results = pool.map(process_file, inputs_for_pool)
                        valid_z = []
                        valid_flux = []
                        valid_snr = []
                        valid_continuum = []
                        valid_specobjid = []
                        valid_filename = []
                        valid_class = []
                        for result in results:
                            if result != (None, None, None, None, None, None, None):
                                valid_z.append(result[0])
                                valid_flux.append(result[1])
                                valid_snr.append(result[2])
                                valid_continuum.append(result[3])
                                valid_specobjid.append(result[4])
                                valid_filename.append(result[5])
                                valid_class.append(result[6])
                            else:
                                total_skipped += 1
                        num_valid = len(valid_z)
                        total_valid += num_valid
                        print(f"Batch valid spectra: {num_valid}")

                        # Append valid results to output datasets if any
                        if num_valid > 0:
                            valid_z = np.array(valid_z)
                            valid_flux = np.array(valid_flux)
                            valid_snr = np.array(valid_snr)
                            valid_continuum = np.array(valid_continuum)
                            valid_specobjid = np.array(valid_specobjid, dtype=str)
                            valid_filename = np.array(valid_filename, dtype=str)
                            valid_class = np.array(valid_class, dtype=str)
                            # Reshape redshift and snr if needed
                            if valid_z.ndim == 1:
                                valid_z = valid_z.reshape(-1, 1)
                            if valid_snr.ndim == 1:
                                valid_snr = valid_snr.reshape(-1, 1)
                            current_rows = dset_flux.shape[0]
                            new_total = current_rows + num_valid
                            dset_flux.resize((new_total, len_data))
                            dset_continuum.resize((new_total, len_data))
                            dset_z.resize((new_total, 1))
                            dset_snr.resize((new_total, 1))
                            dset_specobjid.resize((new_total,))
                            dset_filename.resize((new_total,))
                            dset_class.resize((new_total,))
                            dset_continuum[current_rows:new_total, :] = valid_continuum
                            dset_flux[current_rows:new_total, :] = valid_flux
                            dset_z[current_rows:new_total, :] = valid_z
                            dset_snr[current_rows:new_total, :] = valid_snr
                            dset_specobjid[current_rows:new_total] = valid_specobjid
                            dset_filename[current_rows:new_total] = valid_filename
                            dset_class[current_rows:new_total] = valid_class
                            
                        sys.stdout.flush()

                print("\n--- Processing Completed ---")
                print(f"Total valid spectra processed: {total_valid}")
                print(f"Total spectra skipped: {total_skipped}")

    except FileNotFoundError:
        print(f"Error: Input data file '{raw_data_filepath}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Dataset '{e}' not found in the input HDF5 file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        sys.exit(1)
    with h5py.File(f"data/{preprocessing_config['output_file']}", 'r') as f_in:
        zs = f_in['z_true'][:]
    print(f"\nFinal number of training points saved: z_true: {len(zs)}")

    # --- Cleanup ---
    print("\nPerforming cleanup...")
    config_path = 'config.yaml'
    try:
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"Configuration file '{config_path}' deleted.")
        else:
            print(f"Configuration file '{config_path}' does not exist, skipping deletion.")
    except OSError as e:
        print(f"Error deleting configuration file: {e}")

    print("\n--- Script Execution Finished ---")
