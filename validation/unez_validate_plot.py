import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing
import os
import shutil
import yaml
import sys
import matplotlib.gridspec as gridspec


global len_data
global max_z
global wavelength_template
global means
global columns
global plotting_config


# Load configuration file
config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Load physics information and training config
physics_info = config['physicsinfo']
plotting_config = config['plottinginfo']

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

def width_to_velocity(width):
    center = 6500
    velocity = ((width)/center)*300000

    return velocity


def plot_spectra(filename, z_pred, z_true, flux, continuum, template, decoded, hybrid_loss):  
    try:
    


                
        norm = np.linalg.norm(flux)
        best_shift = redshift_to_shift(z_pred, wavelength_template)  
        true_shift = redshift_to_shift(z_true, wavelength_template)
        flux = flux/norm

        if np.abs((z_pred - z_true) / (1 + z_true)) > 0.0015:
            plt.figure(figsize=(12, 6))
            # Plotting the data
            x_pred = 10**wavelength_template[best_shift:best_shift+len_data]
            predicted_z = z_pred
            true_z = z_true

            plt.plot(x_pred, (flux*norm)+continuum, linewidth=0.6, color='#1C247D',
                    label=f'Input Spectrum; \nTrue z={true_z:.4f}')
            plt.plot(x_pred, (template*norm)+continuum, alpha=1, linewidth=0.5,
                    color='#FF5733', label=f'Predicted Template; \nPredicted z={predicted_z:.4f}\nPredicted v={width_to_velocity(decoded[-1]):.2f} km/s')

            # Set x-axis limits based on x_pred
            x_min = min(x_pred)
            x_max = max(x_pred)
            plt.xlabel('Rest-Frame Wavelength (Å)')

            # Adding vertical lines and labels on the primary axis (predicted rest-frame wavelengths)
            locs = [loc for loc, val in enumerate(means) if 10**val >= x_min and 10**val <= x_max]
            ytops = []
            for step, loc in enumerate(locs):
                ytop = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 1.1*max((flux*norm)+continuum)  + 1
                ytops.append(ytop)
                x_pos = 10**means[loc]  # Predicted rest-frame wavelength
                plt.axvline(x=x_pos, color='#FF5733', linestyle='dashed', linewidth=0.4, alpha=0.5)
                plt.text(x_pos, ytop, columns[loc], color='#FF5733', rotation=0,
                        verticalalignment='bottom', horizontalalignment='center', fontsize=6)

            plt.legend()
            ax1 = plt.gca()

            # Setting up the secondary x-axis
            ax2 = ax1.twiny()
            x1_ticks = ax1.get_xticks()

            def predicted_to_true_rest_wavelength(lambda_pred, predicted_z, true_z):
                return lambda_pred * (1 + predicted_z) / (1 + true_z)

            x2_labels = predicted_to_true_rest_wavelength(x1_ticks, predicted_z, true_z)
            x2_labels_formatted = [f'{val:.2f}' for val in x2_labels]

            ax2.set_xticks(x1_ticks)
            ax2.set_xticklabels(x2_labels_formatted)
            ax2.set_xlabel('True Rest-Frame Wavelength (Å)')
            ax2.set_xlim(ax1.get_xlim())

            # Adding vertical lines and labels for true rest-frame wavelengths adjusted to predicted rest-frame wavelengths
            true_locs = []
            x_positions = []

            for loc, val in enumerate(means):
                lambda_rest_true = 10**val  # True rest-frame wavelength
                # Convert to predicted rest-frame wavelength
                x_pos = lambda_rest_true * (1 + true_z) / (1 + predicted_z)
                if x_min <= x_pos <= x_max:
                    true_locs.append(loc)
                    x_positions.append(x_pos)

            for step, (true_loc, x_pos) in enumerate(zip(true_locs, x_positions)):
                ytop = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 1.1*max((flux*norm)+continuum) + 1
                ybot = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 0.8*min((flux*norm)+continuum) - 3
                plt.axvline(x=x_pos, color='purple', linestyle='dashed', linewidth=0.4, alpha=0.5)
                plt.text(x_pos, ybot, columns[true_loc], color='purple', rotation=0,
                        verticalalignment='bottom', horizontalalignment='center', fontsize=6)

            # Ensure ytop and ybot are defined before using them
            plt.ylim(-0.2*(ytop-ybot) - 3, 1.3*max((flux*norm)+continuum) + 1)

            bruh = filename[-20:-1]

            plt.savefig(f'validation_plots/unez_fits_outliers/{bruh}.pdf', dpi=1000)
            
            
            
            
            plt.figure()
            plt.plot([shift_to_redshift(k, wavelength_template) for k in range(len(wavelength_template)-len_data+1)], hybrid_loss)
            plt.axvline(x=z_pred, color='red', linestyle='dashed')
            plt.axvline(x=z_true, color='blue', linestyle='dashed')
            plt.savefig(f'validation_plots/unez_loss_outliers/{bruh}.pdf', dpi=1000)

            
            
            
            plt.close('all')

        if np.abs((z_pred - z_true) / (1 + z_true)) < 0.0015:
            plt.figure(figsize=(12, 6))
            # Plotting the data
            x_pred = 10**wavelength_template[best_shift:best_shift+len_data]
            predicted_z = shift_to_redshift(best_shift, wavelength_template)
            true_z = z_true

            plt.plot(x_pred, (flux*norm)+continuum, linewidth=0.6, color='#1C247D',
                    label=f'Input Spectrum; \nTrue z={true_z:.4f}')
            plt.plot(x_pred, (template*norm)+continuum, alpha=1, linewidth=0.5,
                    color='#FF5733', label=f'Predicted Template; \nPredicted z={predicted_z:.4f}\nPredicted v={width_to_velocity(decoded[-1]):.2f} km/s')

            # Set x-axis limits based on x_pred
            x_min = min(x_pred)
            x_max = max(x_pred)
            plt.xlabel('Rest-Frame Wavelength (Å)')

            # Adding vertical lines and labels on the primary axis (predicted rest-frame wavelengths)
            locs = [loc for loc, val in enumerate(means) if 10**val >= x_min and 10**val <= x_max]
            ytops = []
            for step, loc in enumerate(locs):
                ytop = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 1.1*max((flux*norm)+continuum)  + 1
                ytops.append(ytop)
                x_pos = 10**means[loc]  # Predicted rest-frame wavelength
                plt.axvline(x=x_pos, color='#FF5733', linestyle='dashed', linewidth=0.4, alpha=0.5)
                plt.text(x_pos, ytop, columns[loc], color='#FF5733', rotation=0,
                        verticalalignment='bottom', horizontalalignment='center', fontsize=6)

            plt.legend()
            ax1 = plt.gca()

            # Setting up the secondary x-axis
            ax2 = ax1.twiny()
            x1_ticks = ax1.get_xticks()

            def predicted_to_true_rest_wavelength(lambda_pred, predicted_z, true_z):
                return lambda_pred * (1 + predicted_z) / (1 + true_z)

            x2_labels = predicted_to_true_rest_wavelength(x1_ticks, predicted_z, true_z)
            x2_labels_formatted = [f'{val:.2f}' for val in x2_labels]

            ax2.set_xticks(x1_ticks)
            ax2.set_xticklabels(x2_labels_formatted)
            ax2.set_xlabel('True Rest-Frame Wavelength (Å)')
            ax2.set_xlim(ax1.get_xlim())

            # Adding vertical lines and labels for true rest-frame wavelengths adjusted to predicted rest-frame wavelengths
            true_locs = []
            x_positions = []

            for loc, val in enumerate(means):
                lambda_rest_true = 10**val  # True rest-frame wavelength
                # Convert to predicted rest-frame wavelength
                x_pos = lambda_rest_true * (1 + true_z) / (1 + predicted_z)
                if x_min <= x_pos <= x_max:
                    true_locs.append(loc)
                    x_positions.append(x_pos)

            for step, (true_loc, x_pos) in enumerate(zip(true_locs, x_positions)):
                ytop = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 1.1*max((flux*norm)+continuum)  + 1

                ybot = 0.1*max((flux*norm)+continuum) * np.sin(1.8 * step) + 0.8*min((flux*norm)+continuum)  -  3
                plt.axvline(x=x_pos, color='purple', linestyle='dashed', linewidth=0.4, alpha=0.5)
                plt.text(x_pos, ybot, columns[true_loc], color='purple', rotation=0,
                        verticalalignment='bottom', horizontalalignment='center', fontsize=6)

            plt.ylim(-0.2*(ytop-ybot) - 3, 1.3*max((flux*norm)+continuum)  + 1)

            bruh = filename[-20:-1]

            plt.savefig(f'validation_plots/unez_fits/{bruh}.pdf', dpi=1000)
            
            
            
            plt.figure()
            plt.plot([shift_to_redshift(k, wavelength_template) for k in range(len(wavelength_template)-len_data+1)], hybrid_loss)
            plt.axvline(x=z_pred, color='red', linestyle='dashed')
            plt.axvline(x=z_true, color='blue', linestyle='dashed')
            plt.savefig(f'validation_plots/unez_loss/{bruh}.pdf', dpi=1000)


            plt.close('all')
        
    except Exception as e:
        print(e)
        print(filename)
        print(z_pred)
        print(z_true)
        print(flux)
        print(continuum)
        print(template)
        print(decoded)
        print('ERROR')
        sys.stdout.flush()
        pass






if __name__ == '__main__':


    print('STARTING SCRIPT')
    sys.stdout.flush()



    if os.path.exists('validation_plots/unez_fits/'):
        shutil.rmtree('validation_plots/unez_fits/')
    if os.path.exists('validation_plots/unez_fits_outliers/'):
        shutil.rmtree('validation_plots/unez_fits_outliers/')
    if os.path.exists('validation_plots/unez_loss/'):
        shutil.rmtree('validation_plots/unez_loss/')
    if os.path.exists('validation_plots/unez_loss_outliers/'):
        shutil.rmtree('validation_plots/unez_loss_outliers/')

    os.makedirs('validation_plots/unez_fits/')
    os.makedirs('validation_plots/unez_fits_outliers')
    os.makedirs('validation_plots/unez_loss/')
    os.makedirs('validation_plots/unez_loss_outliers')




    # Specify the filename
    h5_filename = plotting_config['inference_file']

    files = plotting_config['num_points']

    batch_size = 10000  # Adjust based on available memory

    with h5py.File(h5_filename, 'r') as f:
        # List all datasets in the file
        print("Datasets in the file:")
        for key in f.keys():
            print(key)
        

        # Determine the number of samples to process
        total_records = f['snr_all'].shape[0]
        num_samples = min(total_records, files)
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size else 0)

        print(f"Total samples: {num_samples}, Number of batches: {num_batches}")
        print("Starting batch processing...")
        sys.stdout.flush()

        # Initialize lists to store data for each valid sample
        z_true_all = []
        z_pred_all = []
        decoded_all = []
        filenames_all = []
        snr_all = []

        # Process data in batches
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_samples)
            
            # Read only the current batch for the SNR data
            snr_batch = f['snr_all'][batch_start:batch_end]
            
            # Apply the SNR cut on the batch
            valid_mask = np.array([i for i, n in enumerate(snr_batch) if n > plotting_config['min_snr']])
            
            snr_all.append(snr_batch[valid_mask])
            
            
            valid_count = len(valid_mask)
            print(f"Processing batch {i+1}/{num_batches}: {valid_count} samples passing SNR cut")
            sys.stdout.flush()
            
            # Read the corresponding data from the other datasets using the valid mask
            z_true_all.append(f['z_true_all'][batch_start:batch_end][valid_mask])
            z_pred_all.append(f['z_pred_all'][batch_start:batch_end][valid_mask])
            decoded_all.append(f['decoded_all'][batch_start:batch_end][valid_mask])
            
            # Process filenames and decode them
            filenames_batch = f['filenames_all'][batch_start:batch_end][valid_mask]
            filenames_all.extend([name[0].decode('utf-8') for name in filenames_batch])


        # Concatenate batches into final arrays
        z_true_all = np.concatenate(z_true_all, axis=0)
        z_pred_all = np.concatenate(z_pred_all, axis=0)
        decoded_all = np.concatenate(decoded_all, axis=0)
        snr_all = np.concatenate(snr_all, axis=0)

        print("Batch processing complete!")
        print(f"Final shapes: z_true_all={z_true_all.shape}, z_pred_all={z_pred_all.shape}, decoded_all={decoded_all.shape}")
        sys.stdout.flush()

    # Now you have your arrays back and can use them as needed
    z_nmad = 1.48*np.abs((z_pred_all-z_true_all)/(1+z_true_all))
    z_nmad_mean = np.median(z_nmad[z_nmad > 0.0015])
    outlier_idxs = [i for i, n in enumerate(z_nmad) if n > 0.0015]
    inlier_idxs = [i for i, n in enumerate(z_nmad) if n <= 0.0015]
    outlier_fraction = len(outlier_idxs) / len(z_true_all)

    z_diff = z_true_all - z_pred_all
    mse = []
    pred_templates = []
    cos = []
    losses = []


    print('FILES LOADED \n PLOTTING Z_PRED VS Z_TRUE')
    sys.stdout.flush()



    z_nmad = np.abs((z_pred_all - z_true_all) / (1 + z_true_all))

    # Step 2: Apply a threshold to filter outliers if necessary (e.g., z_nmad < 0.05)
    z_nmad_filtered = z_nmad[z_nmad < 0.0015]

    # Step 3: Compute the median of the filtered residuals
    median_residual = np.median(z_nmad_filtered)

    # Step 4: Compute the deviations from the median
    deviations = np.abs(z_nmad_filtered - median_residual)

    # Step 5: Compute the MAD
    mad = np.median(deviations)

    # Step 6: Compute the NMAD using the scaling factor
    z_nmad_mean = mad / 0.6745  # Alternatively, nmad = mad * 1.4826



    outlier_idxs = [i for i, n in enumerate(z_nmad) if n > 0.0015]
    print([i for i, n in enumerate(z_nmad) if n > 0.05 and z_pred_all[i] < 0.1125 or z_pred_all[i] > 0.115])
    
    sys.stdout.flush()
    outlier_idxs_cleaned = np.where(
        (z_nmad > 0.05) &
        ((z_pred_all < 0.1125) | (z_pred_all > 0.115)) &
        ((z_pred_all < 0.98) | (z_pred_all > 0.995)) &
        ((z_pred_all < 0.871) | (z_pred_all > 0.874)) &
        ((z_pred_all < 0.2580) | (z_pred_all > 0.2590))
    )[0]    
    inlier_idxs = [i for i, n in enumerate(z_nmad) if n <= 0.05]
    outlier_fraction = len(outlier_idxs) / len(z_true_all)

    print(len(outlier_idxs), len(inlier_idxs), len(outlier_idxs_cleaned))
    
    sys.stdout.flush()
    #np.savetxt('sdss_unez_outlier_filenames.txt', np.array(filenames_all)[outlier_idxs], fmt='%s')
    z_diff = np.log10(1+z_true_all) - np.log10(1+z_pred_all)


    decoded_all = decoded_all[inlier_idxs]

    n_ii__o_ii = np.array([(decoded_all[i][31]+decoded_all[i][33])/decoded_all[i][17] for i in range(len(decoded_all))])
    n_ii__ha = np.array([(decoded_all[i][31]+decoded_all[i][33])/decoded_all[i][32] for i in range(len(decoded_all))])
    s_ii__ha = np.array([(decoded_all[i][34]+decoded_all[i][35])/decoded_all[i][32] for i in range(len(decoded_all))])
    o_ii__o_iii = np.array([decoded_all[i][17]/decoded_all[i][27] for i in range(len(decoded_all))])
    y_ratios = np.array([decoded_all[i][27]/decoded_all[i][24] for i in range(len(decoded_all))])

    plt.figure()
    plt.hist(snr_all[outlier_idxs], bins=100, range = (3, 15))
    plt.xlabel('SNR')
    plt.ylabel('Outlier Count')
    plt.savefig('validation_plots/starburst_unez_snr_hist.png', dpi=1000)

    # Plot the aggregated results
    fig = plt.figure(figsize=(10, 10))  # Total figure size
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)  # 4:1 height ratio, small spacing

    # --- Top Plot (Main Redshift Plot) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(np.log10(1+z_true_all), np.log10(1+z_pred_all), 'o', markersize=0.1)
    ax1.set_ylabel(r'$\log_{10}(1 + z_{\mathrm{pred}})$')
    #ax1.set_xlabel('SDSS Catalog Redshift')
    ax1.tick_params(labelbottom=False)  # Hides x-axis tick labels
    ax1.set_xticks([])                 # Removes the x-axis ticks entirely (optional)


    textstr = '\n'.join((
        f'NMAD: {z_nmad_mean:.7f}',
        f'Outlier Fraction: {outlier_fraction:.3f}',
        f'Sample Size: {len(z_true_all)}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax1.text(0.9775, 0.9775, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    #ax1.text(0.2225, 0.9775, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)

    z_true_range = np.linspace(min(z_true_all), max(z_true_all), 1000)
    z_upper = z_true_range + 0.0015 * (1 + z_true_range)
    z_lower = z_true_range - 0.0015 * (1 + z_true_range)
    #ax1.plot(z_true_range, z_upper, 'r--', linewidth=0.5, label='Outlier Boundaries')
    #ax1.plot(z_true_range, z_lower, 'r--', linewidth=0.5)



    # Example x and y data
    # Data
    x = np.array(z_true_all)
    y = np.array(np.log(1+z_true_all) - np.log(1+z_pred_all))
    x = x[:, 0]
    y = y[:, 0]

    print(np.shape(x), np.shape(y))

    # Bins
    x_bins = np.arange(0, 1.0 + 0.00016, 0.00016)
    linear_thresh = 0.0015
    bins_lin = np.linspace(-linear_thresh, linear_thresh, 250)
    bins_log_neg = -np.logspace(np.log10(0.3), np.log10(linear_thresh), 25)
    bins_log_pos = np.logspace(np.log10(0.3), np.log10(linear_thresh), 25)
    y_bins = np.unique(np.concatenate([bins_log_neg, bins_lin, bins_log_pos]))

    # Compute the histogram manually
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Plot
    ax2 = fig.add_subplot(gs[1])

    H_log = np.log10(H + 1e-5)  # Avoid log(0)
    mesh = ax2.pcolormesh(xedges, yedges, H_log.T, cmap='binary')


    #mesh = ax2.pcolormesh(xedges, yedges, H.T, cmap='binary')
    ax2.set_yscale('symlog', linthresh=linear_thresh)
    ax2.axhline(y=linear_thresh, color='#FF5733', linestyle='dashed', linewidth=0.75)
    ax2.axhline(y=-linear_thresh, color='#FF5733', linestyle='dashed', linewidth=0.75)

    ax2.set_xlabel(r'$\log_{10}(1 + z_{\mathrm{SDSS}})$')
    ax2.set_ylabel(r'$\log(1+z_\mathrm{true}) - \log(1+z_\mathrm{pred})$')

    plt.tight_layout()
    plt.savefig('validation_plots/starburst_unez_z_plot.png', dpi=1000)




    print('PLotted Z_PRED VS Z_TRUE \n PLOTTING Z_DIFF HISTOGRAM')
    sys.stdout.flush()

    print(min(z_diff), max(z_diff))
    linear_thresh = 0.0015
    bins_lin = np.linspace(-linear_thresh, linear_thresh, 250)
    bins_log_neg = -np.logspace(np.log10(0.3), np.log10(linear_thresh), 25)
    bins_log_pos = np.logspace(np.log10(0.3), np.log10(linear_thresh), 25)
    bins = np.unique(np.concatenate([bins_log_neg, bins_lin, bins_log_pos]))


    plt.figure()
    #plt.hist(np.log(1+z_true_all) - np.log(1+z_pred_all), bins=bins, label=f'Redshift per pixel: {shift_to_redshift(1, wavelength_template)-shift_to_redshift(0, wavelength_template):.7f}')
    plt.hist(np.log(1+z_true_all) - np.log(1+z_pred_all), bins=bins)
    plt.axvline(x=linear_thresh, color='#FF5733', linestyle='dashed', linewidth=0.75, label = f'Linear threshold')
    plt.axvline(x=-linear_thresh, color='#FF5733', linestyle='dashed', linewidth=0.75)
    plt.xscale('symlog', linthresh=linear_thresh)
    plt.xlabel(r'$\log(1 + z_{\mathrm{true}}) - \log(1 + z_{\mathrm{pred}})$')
    plt.ylabel('Count')
    plt.xticks([-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
   
    textstr = f'Inlier median: {np.mean(np.log(1+z_true_all[inlier_idxs]) - np.log(1+z_pred_all[inlier_idxs])):.7f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = plt.gca()
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)

    
    
    plt.legend()
    plt.savefig('validation_plots/starburst_unez_z_diff_hist.png', dpi=1000)

    def redshift_to_delta_loglam(redshift, wavelength_template):
        return np.log10(1 + redshift)/(wavelength_template[1]-wavelength_template[0]) 
    print('PLOTTING PIXEL DIFFERENCE HISTOGRAM')
    sys.stdout.flush()

    z_diff = np.array(z_diff)

    # Ensure `inlier_idxs` is a NumPy array for fast indexing
    inlier_idxs = np.array(inlier_idxs)

    # Use NumPy indexing instead of a loop
    pixel_diff = redshift_to_delta_loglam(np.abs(z_diff[inlier_idxs]), wavelength_template)

    plt.figure()
    plt.hist(pixel_diff, bins=100)
    plt.xlabel('Pixel Difference')
    plt.ylabel('Count')
    plt.savefig('validation_plots/starburst_unez_pixel_diff_hist.pdf', dpi=1000)

    
    sys.stdout.flush()

    plt.figure()
    plt.plot(np.log10(n_ii__o_ii), np.log10(y_ratios), 'o', markersize =0.5)
    plt.xlabel(f'log({columns[27]}{means[27]}/{columns[24]}{means[24]})')
    plt.ylabel(f'log({columns[31]}{means[31]}/{columns[17]}{means[17]})')
    plt.savefig('validation_plots/starburst_unez_nii_ratios.pdf', dpi=1000)

    plt.figure()
    plt.plot(np.log10(n_ii__ha), np.log10(y_ratios), 'o', markersize =0.5)
    plt.xlabel(f'log({columns[27]}{means[27]}/{columns[32]} {means[32]})')
    plt.ylabel(f'log({columns[31]}{means[31]}/{columns[17]}{means[17]})')
    plt.savefig('validation_plots/starburst_unez_ratios_ha.pdf', dpi=1000)
    #plt.show()

    plt.figure()
    plt.plot(np.log10(s_ii__ha), np.log10(y_ratios), 'o', markersize =0.5)
    #plt.plot(np.arange(-2.5, 1, 0.01), 0.72 / (np.arange(-2.5, 1, 0.01) - 0.32) + 1.30, 'r--', linewidth=0.5)
    plt.xlabel(f'log(({columns[34]}{means[34]}+{columns[35]}{means[35]})/{columns[32]} {means[32]})')
    plt.ylabel(f'log({columns[31]}{means[31]}/{columns[17]}{means[17]})')
    plt.savefig('validation_plots/starburst_unez_ratios_sii.pdf', dpi=1000)

    plt.figure()
    plt.plot(np.log10(o_ii__o_iii), np.log10(y_ratios), 'o', markersize =0.5)
    plt.xlabel(f'log({columns[17]}{means[17]}/{columns[31]}{means[31]})')
    plt.ylabel(f'log({columns[31]}{means[31]}/{columns[17]}{means[17]})')
    plt.savefig('validation_plots/starburst_unez_ratios_oii.pdf', dpi=1000)
    
    means = np.round(10**means).astype(int)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ### First Subplot ###
    mask1 = (n_ii__o_ii > 0) & np.isfinite(n_ii__o_ii) & (y_ratios > 0) & np.isfinite(y_ratios)
    x1 = np.log10(n_ii__o_ii[mask1])
    y1 = np.log10(y_ratios[mask1])

    xlim1 = (-1, 0.5)
    ylim1 = (-2.5, 1)
    nbins = 250
    xedges1 = np.linspace(xlim1[0], xlim1[1], nbins + 1)
    yedges1 = np.linspace(ylim1[0], ylim1[1], nbins + 1)

    H1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=(xedges1, yedges1))
    H1 = H1.T  # Transpose to match the orientation

    axes[1, 0].contourf(xedges1[:-1], yedges1[:-1], H1, levels=10, cmap='binary')
    axes[1, 0].set_xlabel(f'log(({columns[31]}{means[31]}+{columns[33]}{means[33]})/{columns[17]} {means[17]})')
    axes[1, 0].set_ylabel(f'log({columns[27]}{means[27]}/{columns[24]}{means[24]})')
    axes[1, 0].set_xlim(xlim1)
    axes[1, 0].set_ylim(ylim1)
    axes[1, 0].set_title(f'Sample Size = {len(n_ii__o_ii[mask1])}')

    ### Second Subplot ###
    mask2 = (n_ii__ha > 0) & np.isfinite(n_ii__ha) & (y_ratios > 0) & np.isfinite(y_ratios)
    x2 = np.log10(n_ii__ha[mask2])
    y2 = np.log10(y_ratios[mask2])

    xlim2 = (-1, 1)
    ylim2 = (-2.5, 1)
    xedges2 = np.linspace(xlim2[0], xlim2[1], nbins + 1)
    yedges2 = np.linspace(ylim2[0], ylim2[1], nbins + 1)

    H2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=(xedges2, yedges2))
    H2 = H2.T  # Transpose to match the orientation
    
    x_line = np.linspace(-1, 1, 500)
    y_line = 0.61 / (x_line - 0.47) + 1.19
    axes[0, 0].plot(x_line, y_line, 'r--', linewidth=0.5)

    axes[0, 0].contourf(xedges2[:-1], yedges2[:-1], H2, levels=10, cmap='binary')
    axes[0, 0].set_xlabel(f'log(({columns[31]}{means[31]}+{columns[33]}{means[33]})/{columns[32]} {means[32]})')
    axes[0, 0].set_ylabel(f'log({columns[27]}{means[27]}/{columns[24]}{means[24]})')
    axes[0, 0].set_xlim(xlim2)
    axes[0, 0].set_ylim(ylim2)
    axes[0, 0].set_title(f'Sample Size = {len(n_ii__ha[mask2])}')


    ### Third Subplot ###
    mask3 = (s_ii__ha > 0) & np.isfinite(s_ii__ha) & (y_ratios > 0) & np.isfinite(y_ratios)
    x3 = np.log10(s_ii__ha[mask3])
    y3 = np.log10(y_ratios[mask3])

    xlim3 = (-2.5, 0.5)
    ylim3 = (-2.5, 1)
    xedges3 = np.linspace(xlim3[0], xlim3[1], nbins + 1)
    yedges3 = np.linspace(ylim3[0], ylim3[1], nbins + 1)

    H3, xedges3, yedges3 = np.histogram2d(x3, y3, bins=(xedges3, yedges3))
    H3 = H3.T  # Transpose to match the orientation

    axes[0, 1].contourf(xedges3[:-1], yedges3[:-1], H3, levels=10, cmap='binary')
    # Plot the theoretical line
    x_line = np.linspace(-2.5, 1, 500)
    y_line = 0.72 / (x_line - 0.32) + 1.30
    axes[0, 1].plot(x_line, y_line, 'r--', linewidth=0.5)
    axes[0, 1].set_xlabel(f'log(({columns[34]}{means[34]}+{columns[35]}{means[35]})/{columns[32]} {means[32]})')
    axes[0, 1].set_ylabel(f'log({columns[27]}{means[27]}/{columns[24]}{means[24]})')
    axes[0, 1].set_xlim(xlim3)
    axes[0, 1].set_ylim(ylim3)
    axes[0, 1].set_title(f'Sample Size = {len(s_ii__ha[mask3])}')

    ### Fourth Subplot ###
    mask4 = (o_ii__o_iii > 0) & np.isfinite(o_ii__o_iii) & (y_ratios > 0) & np.isfinite(y_ratios)
    x4 = np.log10(o_ii__o_iii[mask4])
    y4 = np.log10(y_ratios[mask4])

    xlim4 = (-1, 2)
    ylim4 = (-2.5, 1)
    xedges4 = np.linspace(xlim4[0], xlim4[1], nbins + 1)
    yedges4 = np.linspace(ylim4[0], ylim4[1], nbins + 1)

    H4, xedges4, yedges4 = np.histogram2d(x4, y4, bins=(xedges4, yedges4))
    H4 = H4.T  # Transpose to match the orientation

    axes[1, 1].contourf(xedges4[:-1], yedges4[:-1], H4, levels=10, cmap='binary')
    axes[1, 1].set_xlabel(f'log({columns[17]}{means[17]}/{columns[27]}{means[27]})')
    axes[1, 1].set_ylabel(f'log({columns[27]}{means[27]}/{columns[24]}{means[24]})')
    axes[1, 1].set_xlim(xlim4)
    axes[1, 1].set_ylim(ylim4)
    axes[1, 1].set_title(f'Sample Size = {len(o_ii__o_iii[mask4])}')

    plt.tight_layout()
    plt.savefig('validation_plots/starburst_unez_combined_ratios.png', dpi=1000)
    #plt.show()

    print(f'Len oii: {len(o_ii__o_iii[mask4])}, Len sii: {len(s_ii__ha[mask3])}, Len nii: {len(n_ii__ha[mask2])}, Len nii: {len(n_ii__o_ii[mask1])}')
    sys.stdout.flush()

    if plotting_config['plot'] == True:


        # Specify the filename
        h5_filename = plotting_config['inference_file']

        files = plotting_config['num_points']

        # Open the HDF5 file in read mode
        with h5py.File(h5_filename, 'r') as f:
            # List all datasets in the file
            print("Datasets in the file:")
            for key in f.keys():
                print(key)
            
            sys.stdout.flush()

            snr_all = f['snr_all'][:files]

            snr_cut_idxs = np.array([i for i, n in enumerate(snr_all) if n > plotting_config['min_snr']])[:plotting_config['num_plotting_points']]


            snr_all = snr_all[snr_cut_idxs]
            # Read datasets into NumPy arrays
            print('reading z')
            sys.stdout.flush()
            z_true_all = f['z_true_all'][snr_cut_idxs].flatten()
            z_pred_all = f['z_pred_all'][snr_cut_idxs].flatten()


            # For datasets saved as (N, 1) arrays, flatten them to 1D arrays
            #snr_all = snr_all.flatten()
            z_true_all = z_true_all.flatten()
            z_pred_all = z_pred_all.flatten()




            print('reading flux')
            sys.stdout.flush()
            flux_all = f['flux_data'][snr_cut_idxs] 

            continuum_all = f['continuum_all'][snr_cut_idxs]
            all_template_data = f['all_template_data'][snr_cut_idxs]
            print('reading encodings')
            sys.stdout.flush()
            
            decoded_all = f['decoded_all'][snr_cut_idxs]
            print('reading filenames')
            sys.stdout.flush()
            
            hybrid_loss_all= f['hybrid_loss'][snr_cut_idxs]
            

            # Read the 'filenames_all' dataset and decode the strings
            filenames_all = f['filenames_all'][snr_cut_idxs]
            # Since 'filenames_all' was saved as an array of arrays, flatten and decode
            filenames_all = [name[0].decode('utf-8') for name in filenames_all]

            """nmad = np.log10(z_true_all/z_pred_all)

            if plot_outliers == True:
                z_idxs = [i for i, n in enumerate ()]"""

        for i in range(len(z_true_all)):
            flux_predicted = all_template_data[i][redshift_to_shift(z_pred_all[i], wavelength_template):redshift_to_shift(z_pred_all[i], wavelength_template)+len_data]
            pred_templates.append(flux_predicted/np.linalg.norm(flux_predicted))
            mse_tmp = np.mean((flux_all[i]/np.linalg.norm(flux_all[i]) - flux_predicted/np.linalg.norm(flux_predicted))**2)
            cosine = np.sum(flux_all[i]*flux_predicted)/(np.linalg.norm(flux_predicted)*np.linalg.norm(flux_all[i]))
            loss = 0.9*mse_tmp + (1-0.9) * -cosine
            mse.append(mse_tmp)
            cos.append(-cosine)
            losses.append(loss)

        z_nmad = 1.48*np.abs((z_pred_all-z_true_all)/(1+z_true_all))
        outlier_idxs = [i for i, n in enumerate(z_nmad) if n > 0.0015]
        inlier_idxs = [i for i, n in enumerate(z_nmad) if n < 0.0015]


        plt.figure()
        plt.plot(np.array(losses)[inlier_idxs], snr_all[inlier_idxs], 'o', markersize=0.1)
        #plt.hist(np.array(losses)[inlier_idxs], bins = 200)
        #plt.savefig('validation_plots/mse_inlier.png')
        
        #plt.figure()
        plt.plot(np.array(losses)[outlier_idxs], snr_all[outlier_idxs], 'o', color='r', markersize=0.1)
        #plt.hist(np.array(losses)[outlier_idxs], bins = 200, alpha = 0.7)
        plt.savefig('validation_plots/mse_outlier.png', dpi=1000)
        
        try:
            multiprocessing.set_start_method('forkserver')  # or 'forkserver'
        except:
            True == True
        #main()

        with multiprocessing.Pool(5) as pool:
            pool.starmap(plot_spectra, zip(np.array(filenames_all), np.array(z_pred_all), np.array(z_true_all), np.array(flux_all), np.array(continuum_all), pred_templates, np.array(decoded_all), np.array(hybrid_loss_all)))

