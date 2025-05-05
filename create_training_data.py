# Import necessary libraries
import h5py  # For interacting with HDF5 files (data storage)
import numpy as np # For numerical operations, especially with arrays
import yaml  # For reading the configuration file

print("--- Script Start ---")

# --- Configuration Loading ---
config_file_path = 'config.yaml' # Path to the experiment configuration
print(f"Loading configuration from: {config_file_path}")
try:
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file) # Load YAML config into a Python dictionary
    print("Configuration loaded successfully.")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_file_path}")
    exit() # Exit if config is missing
except Exception as e:
    print(f"Error loading or parsing configuration file: {e}")
    exit() # Exit on other config errors

# Extract specific configuration sections
try:
    training_config = config['traininginfo']
    preprocessing_config = config['preprocessinginfo']
    physics_config = config['physicsinfo']
except KeyError as e:
    print(f"Error: Missing section in configuration file: {e}")
    exit() # Exit if essential sections are missing

# Extract key parameters for this script
try:
    points = training_config['training_points'] # Desired number of final training samples
    min_snr = training_config['min_snr']       # Minimum acceptable Signal-to-Noise Ratio
    # len_data = physics_config['len_data'] # Length of each data sample (e.g., number of wavelength bins) - Note: This variable is loaded but not used later in this script.
    output_file_preproc = preprocessing_config['output_file'] # Filename of the preprocessed data
    output_file_training = training_config['training_data']   # Filename for the output training data
except KeyError as e:
    print(f"Error: Missing key parameter in configuration file: {e}")
    exit() # Exit if essential parameters are missing

print(f"\nParameters from config:")
print(f" - Target number of training points: {points}")
print(f" - Minimum SNR threshold: {min_snr}")
# print(f" - Expected data length per sample: {len_data}") # Uncomment if len_data is used

# --- Load Preprocessed Data ---
input_hdf5_path = f"preprocessing/data/{output_file_preproc}"
print(f"\nLoading preprocessed data from: {input_hdf5_path}")

try:
    with h5py.File(input_hdf5_path, 'r') as f:
        print("Available datasets in input file:")
        # List all top-level keys in the HDF5 file
        for key in f.keys():
            print(f"- {key} (Shape: {f[key].shape})") # Also print shape for context

        # Load slightly more data than needed initially (factor of 3)
        # This provides a buffer in case many samples are filtered out by the SNR cut.
        num_to_load = 3 * points
        print(f"\nAttempting to load up to {num_to_load} samples initially...")

        # Check if enough data exists before slicing
        available_samples = len(f['snr']) # Use snr as reference for sample count
        if available_samples < num_to_load:
            print(f"Warning: Input file only contains {available_samples} samples, loading all available.")
            num_to_load = available_samples

        # Load the required datasets using slicing
        flux = f['preprocessed_flux'][:num_to_load]
        redshift = f['z_true'][:num_to_load]
        snr = f['snr'][:num_to_load]

        print(f"Loaded initial data shapes:")
        print(f"  Flux: {flux.shape}")
        print(f"  Redshift: {redshift.shape}")
        print(f"  SNR: {snr.shape}")

except FileNotFoundError:
    print(f"Error: Input HDF5 file not found at {input_hdf5_path}")
    exit()
except KeyError as e:
    print(f"Error: Dataset {e} not found in {input_hdf5_path}. Available keys listed above.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the input HDF5 file: {e}")
    exit()

# --- Apply SNR Cut and Select Final Data ---
print(f"\nApplying SNR > {min_snr} cut...")
initial_count = len(snr)
if initial_count == 0:
    print("Error: No data loaded initially. Cannot proceed.")
    exit()

# Create a list of indices for samples meeting the SNR criteria
# This is generally efficient for moderate-sized arrays
snr_mask = [i for i, n in enumerate(snr) if n > min_snr]
# Alternative using numpy boolean indexing (often faster for large arrays):
# snr_mask_np = snr > min_snr
# flux = flux[snr_mask_np]
# redshift = redshift[snr_mask_np]
# snr = snr[snr_mask_np]

# Apply the mask to filter all relevant arrays
flux = flux[snr_mask]
redshift = redshift[snr_mask]
snr = snr[snr_mask] # Filter SNR array itself as well

count_after_snr_cut = len(snr)
print(f"Samples before SNR cut: {initial_count}")
print(f"Samples after SNR cut (> {min_snr}): {count_after_snr_cut}")

# Check if enough samples remain after filtering
if count_after_snr_cut == 0:
    print(f"Error: No samples remaining after applying SNR > {min_snr} cut. Cannot create training file.")
    exit()
elif count_after_snr_cut < points:
    print(f"Warning: Only {count_after_snr_cut} samples remain after SNR cut, which is less than the target {points}. Using all available remaining samples.")
    # No need to change 'points' variable itself, just use the available data.
    # Slicing [:points] below will automatically handle this if count_after_snr_cut < points
else:
    print(f"Sufficient samples remain. Selecting the first {points} samples meeting the SNR criteria.")

# Select the final desired number of samples (or fewer if not enough passed the cut)
flux = flux[:points]
redshift = redshift[:points]
snr = snr[:points]

print(f"\nFinal selected data shapes for training:")
print(f"  Flux: {flux.shape}")
print(f"  Redshift: {redshift.shape}")
print(f"  SNR: {snr.shape}")
final_count = len(flux)
print(f"Total final samples selected: {final_count}")

# --- Save Processed Data for Training ---
output_hdf5_path = f"training/{output_file_training}"
print(f"\nSaving processed data to: {output_hdf5_path}")

try:
    # Open the output file in write mode ('w'). This will overwrite if it exists.
    with h5py.File(output_hdf5_path, 'w') as f:
        # Create datasets in the HDF5 file
        print(f" - Creating dataset 'preprocessed_flux' with shape {flux.shape}")
        f.create_dataset("preprocessed_flux", data=flux)

        print(f" - Creating dataset 'z_true' with shape {redshift.shape}")
        f.create_dataset("z_true", data=redshift)

        print(f" - Creating dataset 'snr' with shape {snr.shape}")
        f.create_dataset("snr", data=snr)

    print("Data saved successfully.")

except Exception as e:
    print(f"An error occurred while writing the output HDF5 file: {e}")
    exit()

print("\n--- Script End ---")