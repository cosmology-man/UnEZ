# UnEZ
Fully unsupervised neural network for redshift and line-strength prediction. 


# Detailed Description
This neural network includes a full pipeline including preprocessing, training, validation, and plotting. The neural network ingests an hdf5 file with all of the raw spectra, wavelength bins for each spectrum, initial redshift predictions for analyzing performance metrics, and identification information such as filenames and specobjid. The model uses only flux and wavelength bins to train the model. The model outputs a redshift prediction, emission/absorption line strength predictions, a global line-width prediction for all lines, and a confidence distribution for all possible redshifts. 
