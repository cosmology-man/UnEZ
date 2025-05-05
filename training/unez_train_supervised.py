import numpy as np
np.seterr(divide='ignore')
import os
os.environ['MPLCONFIGDIR'] = 'tmp/matplotlib'
os.environ['TMPDIR'] = 'tmp'
os.makedirs('tmp/matplotlib', exist_ok=True)
os.makedirs('tmp', exist_ok=True)
import h5py
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.keras.layers import MultiHeadAttention
import sys
from multiprocessing import Pool
import multiprocessing
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
import gc
import math
import yaml
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


#Prime  physics information for model training
global len_data
global max_z
global wavelength_template
global means
global columns
global training_info


# Load configuration file
config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Load physics information and training config
physics_info = config['physicsinfo']
training_info = config['traininginfo']

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

# Fixes any brackets for forbidden lines
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



#set multithreading limits and gpu device
##LEGACY
#tf.config.threading.set_inter_op_parallelism_threads(6)  # For coordinating independent operations
#tf.config.threading.set_intra_op_parallelism_threads(6)  # For speeding up individual operations
gpus = tf.config.experimental.list_physical_devices('GPU')

#set gpu memory growth
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#not in use yet but can be used to potentially distribute training onto multiple gpus
#strategy = tf.distribute.MirroredStrategy()



# Function to adjust column names by appending a counter to duplicates
def adjust_column_names(names):
    counts = {}
    new_names = []
    print(names)
    for name in names:
        if name in counts:
            counts[name] += 1
            new_name = f"{name}.{counts[name]}"
        else:
            counts[name] = 0
            new_name = name
        new_names.append(new_name)
    return new_names







#Simulates starburst galaxy spectra
class data_producer():
    def __init__(self, data_points, min_amplitude, max_amplitude, min_x_value, max_x_value):
        
        #Number of spectra to simulate
        self.data_points = data_points

        #emission/absorption line centers
        self.means = means
        
        #emission/absorption line names
        self.columns = columns
        
        #simulated spectra storage
        self.gaussians_batch = []

        #simulated spectra with noise storage
        self.noisy_gaussians_batch = []

        #redshift converted to pixel shift storage
        self.lambdas = []

        #min/max amplitude of emission/absorption lines for 
        self.min_val = min_amplitude
        self.max_val = max_amplitude

        self.line_strengths = []

        self.dataframe = []

        self.widths = []

        self.noise_spectra = []

        self.snrs = []

        self.wavelength_template = []
        
        
    def initialize_data(self, wavelength_template, vary_height = False, full_line_range = False, true_ratios = True):
        self.wavelength_template = wavelength_template
        means = tf.constant(self.means, dtype = tf.float32)
        """
        
        """
        
        pre_compiled_data = np.zeros((self.data_points, len(self.means)+1))

        if vary_height == False and full_line_range == False:
            for i in range(len(pre_compiled_data)):
                pre_compiled_data[i][18] = 1
                pre_compiled_data[i][19] = 1
                pre_compiled_data[i][24] = 1
                pre_compiled_data[i][25] = 1
                pre_compiled_data[i][32] = 1
                pre_compiled_data[i][33] = 1
                pre_compiled_data[i][34] = 1
                pre_compiled_data[i][-1] = i/pre_compiled_data.shape[0]#18, 19, 25, 24, 33, 34, 32
        
        elif vary_height == True and full_line_range == False:
            for i in range(len(pre_compiled_data)):
                pre_compiled_data[i][18] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][19] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][24] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][25] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][32] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][33] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][34] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][-1] = i/pre_compiled_data.shape[0]#18, 25, 27, 33

        elif vary_height == False and full_line_range == True:
            for i in range(len(pre_compiled_data)):
                used_indices = set()
                while len(used_indices) < 4:
                    random_index = np.random.randint(0, len(self.means))#18, 34)
                    if random_index not in used_indices:
                        pre_compiled_data[i][random_index] = 1
                        used_indices.add(random_index)  # Mark this index as used for this iteration
                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]

        elif vary_height == True and full_line_range == True:
            for i in range(len(pre_compiled_data)):
                used_indices = set()
                while len(used_indices) < 4:
                    random_index = np.random.randint(0, len(self.means))
                    if random_index not in used_indices:
                        pre_compiled_data[i][random_index] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())  # Assign 1 to the unique random index
                        used_indices.add(random_index)  # Mark this index as used for this iteration
                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]
        
        if true_ratios == True:
            
            #initialize ratios
            Ha_Hb = 2.86

            Nii_Ha_max = 10
            Nii_Ha_min = 10**-4

            Oiii_Hb_max = 10
            Oiii_Hb_min = 10**-2.5

            Sii_Ha_max = 10
            Sii_Ha_min = 10**-2.5

            Nii_Oii_max = np.sqrt(10)
            Nii_Oii_min = 10**-1

            Oii_Oiii_max = 100
            Oii_Oiii_min = 10**-1

            

            for i in range(len(pre_compiled_data)):
                
                Nii_Ha = np.random.uniform(Nii_Ha_min, Nii_Ha_max)
                Oiii_Hb = np.random.uniform(Oiii_Hb_min, Oiii_Hb_max)
                Sii_Ha = np.random.uniform(Sii_Ha_min, Sii_Ha_max)
                Nii_Oii = np.random.uniform(Nii_Oii_min, Nii_Oii_max)
                Oii_Oiii = np.random.uniform(Oii_Oiii_min, Oii_Oiii_max)


                Nii_subcontext = Ha_Hb*Nii_Ha/4
                
                Oiii_subcontext = Oiii_Hb

                Sii_coefficient = np.random.uniform(0.2, 2.0)
                Sii_subcontext = Ha_Hb*Sii_Ha/(Sii_coefficient+1)
                Sii_0 = Sii_subcontext*Sii_coefficient
                Sii_1 = Sii_subcontext
                

                
                
                pre_compiled_data[i][24] = 1
                pre_compiled_data[i][32] = Ha_Hb
                pre_compiled_data[i][31] = Nii_subcontext*3
                pre_compiled_data[i][33] = Nii_subcontext
                pre_compiled_data[i][27] = Oiii_subcontext
                pre_compiled_data[i][34] = Sii_0
                pre_compiled_data[i][35] = Sii_1
                pre_compiled_data[i][18] = (Nii_subcontext*3+Nii_subcontext)/Nii_Oii
                



                

                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]

        self.lambdas = pre_compiled_data[:, -1]*redshift_to_shift(0, wavelength_template)

        dataset = tf.data.Dataset.from_tensor_slices(pre_compiled_data).cache()
        dataset = dataset.batch(1024).prefetch(buffer_size=tf.data.AUTOTUNE)

        gaussians_batch = []
        widths = []
        for step, batch in enumerate(dataset):
            tmp_wavelength_template = tf.cast(wavelength_template, dtype=tf.float32)
            batch = tf.cast(batch, dtype=tf.float32)
            widths_tmp = np.random.uniform(1.38, 20, size = len(batch))
            gaussians_batch_tmp = compute_batch_gaussians_tf(tmp_wavelength_template, batch[:, :-1], widths_tmp)
            gaussians_batch_tmp = slice_2d_tensor_by_1d_indices(gaussians_batch_tmp, batch[:, -1]*redshift_to_shift(0, wavelength_template))

            for i, j in zip(gaussians_batch_tmp.numpy(), widths_tmp):
                gaussians_batch.append(i)
                widths.append(j)
            sys.stdout.flush()
        gaussians_batch = np.array(gaussians_batch)

        self.gaussians_batch = gaussians_batch

        adjusted_columns = adjust_column_names(self.columns)

        df = pd.DataFrame(pre_compiled_data, columns=adjusted_columns)

        self.line_strengths = pre_compiled_data[:, :-1]

        print(df)

        self.dataframe = df
        self.widths = np.array(widths).flatten()

        
        return gaussians_batch, self.lambdas, df, np.array(widths).flatten()
    
    def noise_injector(self, snr_min, snr_max, dust_attenuation = True, dust_no_noise = False, plot_dust_curve = True):
        """
        Adds Gaussian noise to each spectrum in the self.spectra array based on a range of signal-to-noise ratios.
        :param snr_min: Minimum signal-to-noise ratio.
        :param snr_max: Maximum signal-to-noise ratio.
        """
        noisy_spectra = np.copy(self.gaussians_batch)
        num_spectra = noisy_spectra.shape[0]

        


        EBVs = np.random.uniform(0, 1.2345, size=(num_spectra))

        A_vs = EBVs*4.05

        for i in range(num_spectra):
            if dust_attenuation == True:
                tau = calzetti_law(10**wavelength_template, EBVs[i])
                tau_z = slice_2d_tensor_by_1d_indices(np.array([tau], dtype=np.float32), np.array([self.dataframe['Z'][i]*redshift_to_shift(0, wavelength_template)], dtype=np.float32))[0]
                noisy_spectra[i] = noisy_spectra[i] / tau_z
                print(self.dataframe['Z'][i]*redshift_to_shift(0, wavelength_template))
                if plot_dust_curve == True:
                    if i == 0:
                        plt.figure('calzetti et al 2000 curve')
                        plt.plot(10**wavelength_template, tau, label = f'$A_{{v}}$ = {A_vs[0]:.2f}')
                        plt.legend()
                        plt.xlabel('Wavelength [$\AA$]')
                        plt.ylabel('$F_{e}/F_{ob}$')
                        




            width = (self.widths[i])
            amps = self.dataframe.iloc[i]#noisy_spectra[i]
            signal = sum_gaussian_areas(amps, width, EBVs[i])


            #snr = np.random.uniform(snr_min, snr_max)
            #noise_std = signal / snr
            noise = np.random.normal(loc=0.0, scale=0.0007, size=noisy_spectra.shape[1])
            self.noise_spectra.append(noise)

            snr = signal/((width*np.sqrt(np.sum(noise**2)/(10**self.wavelength_template[-1]-10**self.wavelength_template[0]))))
            self.snrs.append(snr)
            if dust_no_noise == False:
                noisy_spectra[i] += noise
            
        return noisy_spectra


def calzetti_law(wavelength, EBV, R_V=4.05):
    """
    Calculate the Calzetti law attenuation factor k(lambda) for a given wavelength.
    
    Parameters:
    - wavelength: Wavelength in Angstroms
    - R_V: Total-to-selective extinction ratio (default is 4.05)
    
    Returns:
    - k_lambda: Attenuation factor at the given wavelength
    """
    
    k_lam = np.zeros(len(wavelength))
    lambda_um = wavelength / 10000.0  # Convert to micrometers
    closest_idx = np.abs(lambda_um-0.63).argmin()

    k_lam[closest_idx:] = (2.659 * (-1.857 + 1.040 / lambda_um[closest_idx:]) + R_V)
    k_lam[:closest_idx] = (2.659 * (-2.156 + (1.509 / lambda_um[:closest_idx]) - (0.198 / lambda_um[:closest_idx]**2) + (0.011 / lambda_um[:closest_idx]**3)) + R_V)
    


    ESBV = EBV
    
    tau = 10 ** (0.4 * ESBV * k_lam)


    return tau

def calzetti_law_single(wavelength, EBV, R_V=4.05):
    """
    Calculate the Calzetti law attenuation factor k(lambda) for a given wavelength.
    
    Parameters:
    - wavelength: Wavelength in Angstroms
    - R_V: Total-to-selective extinction ratio (default is 4.05)
    
    Returns:
    - k_lambda: Attenuation factor at the given wavelength
    """
    
    k_lam = 0
    lambda_um = wavelength / 10000.0  # Convert to micrometers

    if lambda_um >= 0.63:
        k_lam = (2.659 * (-1.857 + 1.040 / lambda_um) + R_V)    
    else:    
        k_lam = (2.659 * (-2.156 + (1.509 / lambda_um) - (0.198 / lambda_um**2) + (0.011 / lambda_um**3)) + R_V)    
    


    ESBV = EBV
    
    tau = 10 ** (0.4 * ESBV * k_lam)


    return tau



def sum_gaussian_areas(amplitudes, sigma, EBV):
    """
    Calculate the sum of the areas under multiple Gaussian curves with the same width.

    Parameters:
    amplitudes (list of float): The amplitudes of the Gaussian functions.
    sigma (float): The standard deviation (width) of each Gaussian.

    Returns:
    float: The total area under all the Gaussian curves.
    """
    means = tf.constant([
        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
        5895.6, 8500.36, 8544.44, 8664.52
    ], dtype=tf.float32)


    sqrt_2pi_sigma = np.sqrt(2 * np.pi*sigma)  #Compute sigma*sqrt(2*pi) once since it's common for all
    amplitude = np.sort(amplitudes.values[:-1])[-2]#max(amplitudes)
    index = np.argsort(amplitudes.values)[-2]
    attenuated_signal = amplitude/calzetti_law_single(means[index], EBV)

    total_area = 0.68*attenuated_signal * sqrt_2pi_sigma
    return total_area




def inverted_relu(x):
    return -tf.nn.relu(x)  # Negate the output of the standard ReLU


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
    
    
class GlobalSumPooling1D(Layer):
    def __init__(self):
        super(GlobalSumPooling1D, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


def SpectrumEncoder(input_shape=(len_data, 1), n_latent=10, n_hidden=[128, 64, 32], dropout=0.5):


    input_layer = layers.Input(shape=input_shape)

    # Convolutional blocks
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)


    x = layers.Conv1D(128, kernel_size=11, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = layers.Conv1D(256, kernel_size=21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)


    y = layers.Conv1D(128, kernel_size=64, padding='same', activation='relu')(x)
    y = layers.BatchNormalization()(y)
    y = Dropout(0.2)(y)
    
    y = layers.Conv1D(49, kernel_size=64, padding='same', activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = Dropout(0.2)(y)


    z = layers.Conv1D(256, kernel_size=64, padding='same', activation='relu')(x)
    z = layers.BatchNormalization()(z)
    z = Dropout(0.2)(z)

    z = layers.Conv1D(512, kernel_size=64, padding='same', activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = Dropout(0.2)(z)


    x = MultiHeadAttention(num_heads=4, key_dim=32 // 4)(y, z)
    x = Dropout(0.1)(x)


    x = GlobalSumPooling1D()(x)
    
    # Different specialized outputs
    sigmoid_part = Dense(1, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(np.log10(1.01)))(x)  # Use linear here because ScaledSigmoid applies the sigmoid
    sigmoid_part = ScaledSigmoid(min_val=1.38, max_val=20)(sigmoid_part)

    decoded_collection = []

    activation_mapping = {
        -1: inverted_relu,
        0: 'linear',
        1: 'relu'
    }

    outputs = []
    for i, code in enumerate(physics_info['activations']):
        print(code)
        activation_fn = activation_mapping[code]
        # Use different unit sizes if needed; here we set a constant as an example.
        dense_out = Dense(1, activation=activation_fn)(x)
        outputs.append(dense_out)
    
    outputs.append(sigmoid_part)
    


    # Concatenating the two parts back together
    decoded = Concatenate()(outputs)

    model = Model(inputs=input_layer, outputs=decoded)
    return model

    
def build_model():
    model = SpectrumEncoder()

    model.summary()
    return model


def build_model_finetune():
    def inverted_relu(x):
        return -tf.nn.relu(x)  # Negate the output of the standard ReLU

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
        
        
    class GlobalSumPooling1D(Layer):
        def __init__(self, **kwargs):
            super(GlobalSumPooling1D, self).__init__()

        def call(self, inputs):
            return tf.reduce_sum(inputs, axis=1)

    model = load_model('all_object_model.keras', custom_objects = {'ScaledSigmoid': ScaledSigmoid,
                                                                    'GlobalSumPooling1D': GlobalSumPooling1D,
                                                                    'inverted_relu': inverted_relu})
    model.summary()
    return model


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


#@tf.function
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



@tf.function
def find_min_euclidean_distance_index(large_arrays, tiny_arrays,
                                      alpha=1.0, k=25, radius=1.0):
    """
    Align each row of 'large_arrays' (template) with each row of 'tiny_arrays' (spectrum),
    returning:
      1) best_match_indices: index of the best match (lowest hybrid loss) in each row
      2) loss: the final aggregated scalar
      3) hybrid_loss: per-row, per-window array of the hybrid loss
    while preserving EXACT outputs from your original code.
    """

    # Ensure data types are consistent
    large_arrays = tf.cast(large_arrays, dtype=tf.float32)
    tiny_arrays  = tf.cast(tiny_arrays,  dtype=tf.float32)

    # ---------------------------------------------------------
    # 1) Process each row individually via tf.map_fn
    # ---------------------------------------------------------
    def _process_single_pair(inputs):
        """
        Given one row from large_arrays and one row from tiny_arrays, do exactly
        the same gather + MSE + dot-product + norms steps as in the original code
        for that single row.
        """
        large_array_i = inputs[0]  # shape [large_length_i]
        tiny_array_i  = inputs[1]  # shape [tiny_length_i]

        large_length_i = tf.shape(large_array_i)[0]
        tiny_length_i  = tf.shape(tiny_array_i)[0]

        # Number of valid sliding windows
        num_windows_i = large_length_i - tiny_length_i + 1

        # Build the "sliding window" indices
        #  shape => [tiny_length_i, num_windows_i]
        window_indices_i = (
            tf.expand_dims(tf.range(num_windows_i), 0)
            + tf.expand_dims(tf.range(tiny_length_i), 1)
        )

        # Gather those windows
        #  shape => [tiny_length_i, num_windows_i]
        large_windows_i = tf.gather(large_array_i, window_indices_i)

        # Expand tiny_array_i to [tiny_length_i, 1] for broadcast ops
        #  shape => [tiny_length_i, 1]
        tiny_expanded_i = tf.expand_dims(tiny_array_i, axis=1)

        # ----- MSE portion -----
        # squared diff => shape [tiny_length_i, num_windows_i]
        squared_diff_i = tf.square(large_windows_i - tiny_expanded_i)
        # MSE => mean over tiny_length_i => shape [num_windows_i]
        mse_i = tf.reduce_mean(squared_diff_i, axis=0)

        # ----- Dot products and norms for cosine similarity -----
        # dot_products => shape [num_windows_i]
        dot_products_i = tf.reduce_sum(large_windows_i * tiny_expanded_i, axis=0)
        # norm of each window => shape [num_windows_i]
        norm_large_i = tf.norm(large_windows_i, axis=0)
        # Return everything needed for final hybrid_loss
        return mse_i, dot_products_i, norm_large_i

    # Map over rows in parallel
    #   parallel_iterations => let TensorFlow run many rows in parallel.
    #   Increase or decrease as needed based on your GPU/CPU environment.
    (mse, dot_products, norm_large) = tf.map_fn(
        fn=_process_single_pair,
        elems=(large_arrays, tiny_arrays),
        dtype=(tf.float32, tf.float32, tf.float32),
        parallel_iterations=64  # or another suitably large number
    )

    # Now mse, dot_products, norm_large each has shape => [batch_size, num_windows]
    # norm of each row in 'tiny_arrays' => shape [batch_size, 1]
    norm_tiny = tf.norm(tiny_arrays, axis=1, keepdims=True)

    # Cosine similarities => shape [batch_size, num_windows]
    cosine_similarities = dot_products / (norm_large * norm_tiny)

    # Hybrid loss => shape [batch_size, num_windows]
    # alpha * MSE + (1 - alpha) * -cosine_similarity
    hybrid_loss = alpha * mse + (1.0 - alpha) * -cosine_similarities

    # ---------------------------------------------------------
    # 2) Find the top k best matches for each row
    # ---------------------------------------------------------
    # tf.nn.top_k expects to pick largest values,
    # so we pass -hybrid_loss to pick top k smallest
    values, indices = tf.nn.top_k(-hybrid_loss, k, sorted=True)
    # Convert back to positive (lowest) values
    values = -values

    # Weighted average of top k
    weights = tf.exp(-tf.range(k, dtype=tf.float32) / radius)
    weights = weights / tf.reduce_sum(weights)  # normalize
    weighted_top_k_avg = tf.reduce_sum(values * weights, axis=1)
    loss = tf.reduce_mean(weighted_top_k_avg)   # final scalar

    # ---------------------------------------------------------
    # 3) For each row, pick the argmin of hybrid_loss
    # ---------------------------------------------------------
    best_match_indices = tf.argmin(hybrid_loss, axis=1)

    # Return exactly the same three outputs
    return best_match_indices, loss, hybrid_loss



def find_min_euclidean_distance_index_with_fraction(
    large_arrays,
    tiny_arrays,
    alpha=1.0,
    fraction=0.9,   # e.g. keep windows that together contribute at least 90% of the total importance
    # (Unlike before, where fraction might have been 0.3 to mean top 30% of windows by count)
    radius=1.0     # Retained for consistency; might be used elsewhere
):
    """
    Align each row of 'large_arrays' (template) with each row of 'tiny_arrays' (spectrum),
    returning:
      1) best_match_indices: index of the best match (lowest hybrid loss) in each row
      2) loss: the final aggregated scalar loss
      3) hybrid_loss: per-row, per-window array of the hybrid loss

    This variant replaces the fixed-count (or fixed relative threshold) approach with one that
    is inspired by PCA: first, every window is assigned an importance value such that all
    importances sum to 1. Then the windows are sorted in descending order by importance.
    The minimal number of windows that together account for at least the provided fraction
    (e.g. 90% of the total "importance") is chosen, and their losses are aggregated via a weighted
    average (with weights re-normalized).
    """
    import tensorflow as tf

    # Cast arrays to float32 for compatibility.
    large_arrays = tf.cast(large_arrays, dtype=tf.float32)
    tiny_arrays  = tf.cast(tiny_arrays,  dtype=tf.float32)

    # ---------------------------------------------------------
    # 1) Process each row individually via tf.map_fn
    # ---------------------------------------------------------
    def _process_single_pair(inputs):
        large_array_i = inputs[0]  # shape [large_length_i]
        tiny_array_i  = inputs[1]  # shape [tiny_length_i]

        large_length_i = tf.shape(large_array_i)[0]
        tiny_length_i  = tf.shape(tiny_array_i)[0]

        # Number of valid sliding windows.
        num_windows_i = large_length_i - tiny_length_i + 1

        # Build sliding window indices:
        #   shape: [tiny_length_i, num_windows_i]
        window_indices_i = (
            tf.expand_dims(tf.range(num_windows_i), 0)
            + tf.expand_dims(tf.range(tiny_length_i), 1)
        )

        # Gather windows:
        #   shape: [tiny_length_i, num_windows_i]
        large_windows_i = tf.gather(large_array_i, window_indices_i)

        # Expand tiny_array_i for broadcasting:
        #   shape: [tiny_length_i, 1]
        tiny_expanded_i = tf.expand_dims(tiny_array_i, axis=1)

        # ----- MSE calculation -----
        squared_diff_i = tf.square(large_windows_i - tiny_expanded_i)
        mse_i = tf.reduce_mean(squared_diff_i, axis=0)

        # ----- Cosine similarity components -----
        dot_products_i = tf.reduce_sum(large_windows_i * tiny_expanded_i, axis=0)
        norm_large_i = tf.norm(large_windows_i, axis=0)

        return mse_i, dot_products_i, norm_large_i

    (mse, dot_products, norm_large) = tf.map_fn(
        fn=_process_single_pair,
        elems=(large_arrays, tiny_arrays),
        dtype=(tf.float32, tf.float32, tf.float32),
        parallel_iterations=64
    )

    # norm_tiny is computed per row in tiny_arrays.
    norm_tiny = tf.norm(tiny_arrays, axis=1, keepdims=True)

    # Compute cosine similarities.
    cosine_similarities = dot_products / (norm_large * norm_tiny)

    # Hybrid loss for each window.
    hybrid_loss = alpha * mse + (1.0 - alpha) * -cosine_similarities

    # ---------------------------------------------------------
    # 2) PCA-Inspired Top-Fraction Aggregation per Row
    # ---------------------------------------------------------
    def _top_fraction_per_row(row_loss):
        """
        For a given row_loss (shape: [num_windows]),
        compute the importance of each window as a probability distribution (summing to 1)
        by applying a softmax over negative loss values. Lower loss implies higher importance.
        
        Then, sort the windows in descending order by importance and compute the cumulative
        sum of importance. Find the smallest set of top windows which together account for
        at least the specified fraction (e.g. 90%) of the total importance. Aggregate the losses
        of those windows as a weighted average, using the normalized importance as weights.
        """
        # Compute importance using softmax so that sum(importance) == 1.
        # Since we use -row_loss, lower loss values yield higher probabilities.
        importance = tf.nn.softmax(-row_loss)  # shape: [num_windows]

        # Sort windows by importance in descending order.
        sorted_indices = tf.argsort(importance, direction='DESCENDING')
        sorted_importance = tf.gather(importance, sorted_indices)
        sorted_losses = tf.gather(row_loss, sorted_indices)

        # Compute cumulative importance.
        cumulative_importance = tf.cumsum(sorted_importance)

        # Identify the smallest index where the cumulative sum exceeds or equals the fraction threshold.
        valid_indices = tf.where(cumulative_importance >= fraction)
        # Since cumulative_importance[-1] == 1, valid_indices should never be empty.
        k_i = tf.cond(
            tf.size(valid_indices) > 0,
            lambda: valid_indices[0, 0] + 1,  # +1 since indices are 0-based (we need count)
            lambda: tf.shape(row_loss)[0]
        )

        # Re-normalize the importance weights of the selected windows to sum to 1.
        selected_importance = sorted_importance[:k_i]
        normalized_importance = selected_importance / tf.reduce_sum(selected_importance)
        print(k_i)
        sys.stdout.flush()
        # Compute the weighted average of the corresponding losses.
        weighted_loss = tf.reduce_sum(sorted_losses[:k_i] * normalized_importance)

        return weighted_loss

    top_fraction_losses = tf.map_fn(
        fn=_top_fraction_per_row,
        elems=hybrid_loss,
        dtype=tf.float32
    )

    # The overall loss is the mean aggregated loss over all rows.
    loss = tf.reduce_mean(top_fraction_losses)

    # ---------------------------------------------------------
    # 3) For each row, pick the window with the minimal hybrid loss as "best match."
    # ---------------------------------------------------------
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
    return initial_radius * (decay_rate ** -epoch)



class unez():
    def __init__(self, learning_rate, wavelength_template, top_k_values = 1900):

        self.learning_rate = learning_rate
        self.opt = Adam(learning_rate=self.learning_rate)#, clipnorm=1)

        if training_info['finetune'] == True:
            self.autoencoder = build_model_finetune()
            
        else:
            self.autoencoder = build_model()

        self.emissions = [
                            1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
                            1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
                            1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
                            3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
                            4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
                            4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
                            6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
                            6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
                            5895.6, 8500.36, 8544.44, 8664.52
                            ]
        
        self.wavelength_template = tf.constant(wavelength_template, dtype = tf.float32)
        self.top_k = top_k_values

    #single training step
    def pretrain_step(self, batch_data, lammies, alpha, lambdas, r):
        
        #begin gradient recording
        with tf.GradientTape(persistent=True) as tape:

            #get the decoded values from the autoencoder
            decoded = self.autoencoder(batch_data, training=True)  # [batch_size, len(numbers)]
            
            #cast the shifts to float32
            lammies = tf.cast(lammies, dtype=tf.float32)
            
            #compute full range gaussian template
            gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])
            
            #compute normalization factor
            gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
            
            #normalize full range gaussian template to sum to 1
            gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

            #Calculate best shift values, top-k loss, and full range hybrid loss
            best_starting_lambdas, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data, alpha = alpha, radius = r)
            #best_starting_lambdas, test_loss, hybrid_loss = find_min_euclidean_distance_index_with_fraction(gaussians_batch_full, batch_data, alpha = alpha, radius = r, fraction = 0.05)

            #slice the full range gaussian template to the best starting shifts
            gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast(best_starting_lambdas, dtype = tf.float32))   
            
            #causing errors, come back and fix this later
            true_lammy_loss = 0#tf.reduce_mean(tf.square(tf.cast(lammies, dtype = tf.float32)-tf.cast(best_starting_lambdas, dtype = tf.float32)))
            
            #calculate average difference between true shifts and best starting shifts
            delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies, dtype=tf.float32)- tf.cast(best_starting_lambdas, dtype=tf.float32))))

            batch_size = tf.shape(hybrid_loss)[0]
            tmp_loss = tf.zeros_like(hybrid_loss, dtype=tf.float32)

            # Build indices: [[0, lammies[0]], [1, lammies[1]], ...]
            indices = tf.stack([tf.range(batch_size), tf.cast(lammies, tf.int32)], axis=1)

            # Values to assign
            updates = tf.fill([batch_size], -0.1)

            # Scatter update
            tmp_loss = tf.tensor_scatter_nd_update(tmp_loss, indices, updates)
                      
            mse_tmp_vs_hybrid = tf.reduce_mean(tf.square(hybrid_loss - tmp_loss))
      
            loss = tf.reduce_mean(test_loss) + mse_tmp_vs_hybrid

            #loss = (tf.reduce_mean(test_loss))

        #calculate gradients
        autoencoder_gradients = tape.gradient(loss, self.autoencoder.trainable_variables)

        #zip gradients and their corresponding trainable variables
        grads_and_vars = zip(autoencoder_gradients, self.autoencoder.trainable_variables)

        #apply the gradients to trainable variables
        self.opt.apply_gradients(grads_and_vars)

        #delete variables to free up memory
        del lammies, decoded
        gc.collect()


        return loss, gaussians_batch, delta_lam, hybrid_loss, best_starting_lambdas, true_lammy_loss


    #calculate validation step
    def validation_step(self, batch_data, shifts, r, batch_size=16, alpha = training_info['alpha']):

        #transfer data to gpu
        dataset = tf.data.Dataset.from_tensor_slices((batch_data, shifts))

        #split data into batches
        dataset = dataset.batch(batch_size)

        #initialize lists to store data
        gaussians_batch_accumulated = []
        gaussians_batch_full_accumulated = []
        true_shift_loss_accumulated = []
        best_shifts_accumulated = []
        true_decoded_accumulated = []
        delta_lam_accumulated = []
        accumulated_loss = []
        accumulated_hybrid_loss = []

        #iterate through batches of data
        for batch_data_segment, shift_segment in dataset:

            #get the decoded values from the autoencoder
            decoded = self.autoencoder.predict(batch_data_segment, verbose = 0)  # [batch_size, len(numbers)]

            #cast the shifts to float32
            lammies_tmp = tf.cast(shift_segment, dtype=tf.float32)
            
            #compupte full range gaussian template
            gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])

            #normalize full range gaussian template to sum to 1
            gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
            gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

            #calculate best shift values, top-k loss, and full range hybrid loss
            #best_shifts, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data_segment, radius = r, alpha = alpha)
            best_shifts, test_loss, hybrid_loss = find_min_euclidean_distance_index_with_fraction(gaussians_batch_full, batch_data_segment, radius = r, alpha = alpha, fraction = 0.01)

            #slice the full range gaussian template by the best starting shifts
            gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast((best_shifts), dtype = tf.float32))

            #calculate average difference between true shifts and best starting shifts
            delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies_tmp, dtype=tf.float32)- tf.cast(best_shifts, dtype=tf.float32))))

            
            #sum the loss values for the batch
            loss = tf.reduce_mean(test_loss)
            
            #accumulate the batch output values
            accumulated_loss.append(loss.numpy())
            delta_lam_accumulated.append(delta_lam.numpy())
            for i in range(len(decoded[:, -1])):
                accumulated_hybrid_loss.append(hybrid_loss[i].numpy())
                gaussians_batch_accumulated.append(gaussians_batch[i].numpy())
                true_decoded_accumulated.append(decoded[i])
                best_shifts_accumulated.append(best_shifts.numpy()[i])
                gaussians_batch_full_accumulated.append(gaussians_batch_full.numpy()[i])

            delta_lam = tf.reduce_mean(delta_lam_accumulated).numpy()

        return gaussians_batch_accumulated,  best_shifts_accumulated, gaussians_batch_full_accumulated, np.array(true_decoded_accumulated), delta_lam, accumulated_loss, accumulated_hybrid_loss



    def train_and_evaluate(self, data, validation, shifts, validation_shifts, epochs=200, batch_size=16):
        
        """
        Trains an autoencoder on the provided data, evaluates its training loss, and returns the encoder model.
        """
        
        #transfer data to gpu
        dataset = tf.data.Dataset.from_tensor_slices((data, shifts))
        dataset = dataset.shuffle(buffer_size=len(data), reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        #initialize alpha and loss lists for plotting
        alpha = training_info['alpha']
        collect_all_loss = []
        collect_all_validation_loss = []
        collect_all_outliers = []

        #initialize training radius
        r_0 = training_info['r_0']

        #begin training loop
        for epoch in range(epochs):
            r = exponential_decay_radius(epoch, r_0, 0.005, 15)
            

            #initialize time for measuring epoch runtime
            start_time = time.time()

            #initialize lists to store evaluation data for the batch
            collect_shift_loss = []
            collect_reconstruction_loss = []
            collect_true_shift_loss = []
            collect_delta_lam = []
            collect_loss = []

            #iterate through batches of data
            for step, (batch_data, batch_shifts) in enumerate(dataset):
                
                #run training step on batch
                loss, gaussians_batch, delta_lam, hybrid_loss, best_starting_shifts, true_shift_loss = self.pretrain_step(batch_data, batch_shifts, alpha, self.top_k, r)   

                #accumulate the batch evaluation values
                collect_delta_lam.append(delta_lam)
                collect_true_shift_loss.append(true_shift_loss)
                collect_loss.append(loss)
            
            #calculate average values for the epoch
            true_shift_loss = np.mean(collect_true_shift_loss)
            loss = np.mean(collect_loss)
            collect_all_loss.append(loss)
            delta_lam = np.mean(collect_delta_lam)

            #print epoch evaluation values
            print(f'\nEPOCH:{epoch}\n TRUE LAMMY LOSS: {true_shift_loss}\n DELTA_LAM: {delta_lam}\n LOSS: {loss}\n')
            
            #flush output print buffer
            sys.stdout.flush()

            #run validation step
            valid_gaussians_batch, valid_best_shifts, valid_full_gaussians, true_valid_decoded, valid_delta_lam, validation_loss, validation_hybrid_loss = self.validation_step(validation, validation_shifts, r)

            #accumulate validation loss 
            collect_all_validation_loss.append(np.mean(validation_loss))

            #print validation evaluation values
            lr = self.opt.learning_rate.numpy()
            print(f'\nVALIDATION:\n VALID DELTA LAM: {valid_delta_lam}\n ADAM LEARNING RATE: {lr}\n ')
            
            #initialize model save path
            

            #save model
            model_name = training_info['model_name']
            self.autoencoder.save(f'models/{model_name}_{epoch}.keras')
            print(f"Autoencoder Model saved to auto_ez_prototype.keras")













            #this part is a mess. i need to attend to it.
            valid_best_shifts_plot = np.array(valid_best_shifts)
            validation_shifts_plot = np.round(np.array(validation_shifts, dtype = np.int32))

            wavelength_template_temp = self.wavelength_template.numpy()

            z_pred = np.array([shift_to_redshift(i, wavelength_template) for i in valid_best_shifts_plot])
            z_true = np.array([shift_to_redshift(i, wavelength_template) for i in validation_shifts_plot])
            z_slope, z_intercept = np.polyfit(z_true, z_pred, 1)
            y_predicted = z_slope * z_true + z_intercept
            residuals = z_pred - y_predicted


            z_outlier = np.abs([i for i in (z_pred-z_true)/(1+z_true)])
            z_nmad = np.mean(z_outlier[z_outlier<0.15])
            outlier_idxs = [i for i, n in enumerate(z_outlier) if n>0.15]
            num_outliers = len(outlier_idxs)

            print(f'Number of outliers: {len(outlier_idxs)}')
            results = {
                'learning_rate': [lr], 'batch_size': [batch_size], 'epochs': [epoch],
                'outliers': [len(outlier_idxs)], 'NMAD': [z_nmad]}
            
            collect_all_outliers.append(num_outliers)   

            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv('hyperparameter_tuning_results.csv', index=False)
            del df

            #plot loss
            plt.figure()
            plt.plot(collect_all_validation_loss, label = 'validation loss')
            plt.plot(collect_all_loss, label = 'training loss')
            plt.legend()
            plt.savefig('training_plots/loss_plot.pdf')

            #plot outlier count
            plt.figure()
            plt.plot(collect_all_outliers, label = 'outliers')
            plt.xlabel('epoch')
            plt.ylabel('number of outliers')
            plt.savefig('training_plots/outliers.pdf')

            
            plot_epoch(epoch, wavelength_template_temp, batch_shifts, best_starting_shifts, batch_data,
                    validation, outlier_idxs, z_true, z_pred, true_valid_decoded, self.emissions, valid_best_shifts_plot,
                    validation_shifts, valid_best_shifts, z_outlier, z_nmad, residuals, validation_hybrid_loss, gaussians_batch, valid_gaussians_batch,
                    valid_full_gaussians)

                    

                
    
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"EPOCH RUNTIME: {execution_time} s")
            sys.stdout.flush()
            del loss, gaussians_batch, true_shift_loss, valid_gaussians_batch

        
        print(f"Autoencoder Model saved to {model_save_path}")

        return self.autoencoder
        




def width_to_velocity(width):
    center = 6500
    velocity = ((width)/center)*300000

    return velocity



def plot_epoch(epoch, wavelength_template_temp, shifts, best_starting_shifts, batch_data, validation, outlier_idxs, 
         z_true, z_pred, true_valid_decoded, emissions, valid_best_shifts_plot, validation_shifts, valid_best_shifts, 
         z_outlier, z_nmad, residuals, validation_hybrid_loss, gaussians_batch, valid_gaussians_batch, valid_full_gaussians):




            ##TRAINING PLOT##
            plt.figure()
            plt.plot(10**(wavelength_template_temp[np.array(shifts[0], dtype = np.int32): np.array(shifts[0], dtype = np.int32)+len(batch_data[0])]), batch_data[0])
            plt.plot(10**(wavelength_template_temp[np.array(best_starting_shifts[0], dtype = np.int32): np.array(best_starting_shifts[0], dtype = np.int32)+len(batch_data[0])]), gaussians_batch[0], alpha = 0.8)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            plt.savefig(f'training_plots/trained_spectra/emission_line{epoch}.pdf')
            

            
            
            ##TRUE VS PREDICTED REDSHIFT##

            fig = plt.figure(figsize=(8, 6))  # Create a figure object

            # Create a GridSpec with 3 rows and 1 column, but the first row takes up 2/3 of the space
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

            # First subplot (2/3 of the figure height)
            ax1 = fig.add_subplot(gs[0:2, 0])
            ax1.set_ylabel('Predicted Redshift')
            ax1.set_xlabel('True Redshift')
            ax1.plot(z_true, z_pred, 'o', markersize=0.5)
            textstr = f'Mean NMAD: {z_nmad}'
            #'\n'.join(())
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.015, 0.985, textstr, transform=plt.gca().transAxes, fontsize=6,
                    verticalalignment='top', horizontalalignment='left', bbox=props)

            # Second subplot (1/3 of the figure height)
            ax2 = fig.add_subplot(gs[2, 0])
            ax2.scatter(z_true[residuals<0.15], residuals[residuals<0.15], color='red', marker='o', s=0.5)  # 's' is the marker size
            ax2.axhline(y=0, color='black', linestyle='--')  # Adds a horizontal line at zero for reference
            ax2.set_xlabel('True Redshift')
            ax2.set_ylabel('Residuals')

            # Adjust layout
            plt.tight_layout()

            # Save the figure
            plt.savefig(f'training_plots/z_plots/z_plot{epoch}.pdf')


            ##  Redshift vs outlier hist  ##
            plt.figure()
            plt.hist(z_true[outlier_idxs], bins = 70)
            plt.xlabel('Redshift')
            plt.ylabel('Outlier Count')
            plt.savefig(f'training_plots/redshift_outlier_hist/redshift_outlier_{epoch}.pdf')



            
          


            ## WIDTHS PLOT ##
            plt.figure()
            plt.hist(np.array([width_to_velocity(true_valid_decoded[i, -1]) for i in range(len(true_valid_decoded))]), bins = 100)
            plt.xlabel('Predicted Velocity')
            plt.ylabel('Count')
            plt.savefig(f'training_plots/widths_plots/predicted_width_plot{epoch}.pdf')




if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')  # or 'forkserver'
    except:
        True == True

    points = training_info['training_points']
    with h5py.File(training_info['training_data'], 'r') as f:
        for key in f.keys():
            print(f"- {key}")
        sys.stdout.flush()
        z = f['z_true'][:]
        flux = f['preprocessed_flux'][:]
        snr = f['snr'][:]
    
    data_points = len(z)
    
    noisy_norms = np.linalg.norm(flux, axis=1, keepdims=True)

    noisy_data = flux/noisy_norms


    lambdas = np.array([redshift_to_shift(i, wavelength_template) for i in z])
    print(f'Number of redshifts: {np.shape(z)}')
    print(f'Number of spectra: {np.shape(flux)}')
    

    train_lambdas = lambdas[:int(data_points*5/6)]
    train_data = noisy_data[:int(data_points*5/6)]

    validation_lambdas = lambdas[int(data_points*5/6):]
    validation_data = noisy_data[int(data_points*5/6):]


    unez_model = unez(training_info['learning_rate'], wavelength_template, top_k_values = training_info['top_k'])

    trained_unez = unez_model.train_and_evaluate(train_data, validation_data, train_lambdas, validation_lambdas,
                                           epochs = training_info['epochs'], batch_size = training_info['batch_size'])
    
 
