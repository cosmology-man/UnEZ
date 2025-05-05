import os
import shutil
import sys
import subprocess
from datetime import datetime
import argparse
import yaml



config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Load config file
preprocessing_config = config['preprocessinginfo']


if __name__ == "__main__":    
    # Copy the config file to the validation directory if it exists
    run_txt_path = os.path.join(os.getcwd(), 'config.yaml')
    if os.path.exists(run_txt_path):
        shutil.copy(run_txt_path, 'validation/')
    
    """preprocessing_file = f"preprocessing/data/{preprocessing_config['output_file']}"
    if os.path.exists(preprocessing_file):
        shutil.copy(preprocessing_file, 'validation/')
    else:
        print(f'File does not exist: {preprocessing_file}')"""
    
    # Run the main script in the new run directory
    os.chdir('validation/')
    command = "sbatch unez_validate.sh"
    subprocess.run(command, shell=True, check=True)
