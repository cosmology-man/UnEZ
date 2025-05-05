import os
import shutil
import sys
import subprocess
from datetime import datetime
import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a script in a new directory.')
    parser.add_argument('--run_name', type=str, required=True, help='The name of the run.')
    return parser.parse_args()


def create_run_directory(parent_dir, run_name):
    """Create a run directory and copy the folder structure and selected files to it.

    Only the directory structure is recreated for subdirectories.
    For the base directory, only files listed in "files_to_copy.txt" are copied.

    Args:
        parent_dir (str): The parent directory where the run directory will be created.
        run_name (str): The name of the run directory.

    Returns:
        str: The path of the created run directory.
    """

    # Create the parent directory for all runs if it doesn't exist
    os.makedirs(parent_dir, exist_ok=True)

    # Create the main run directory
    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    dirs = ['training_plots', 'training_plots/snr_plots', 'training_plots/redshift_outlier_hist', 
            'training_plots/trained_spectra', 'training_plots/widths_plots', 
            'training_plots/z_plots', 'training_plots/widths_snr']
    
    for dir in dirs:
        dest_dir = os.path.join(run_dir, dir)
        os.makedirs(dest_dir, exist_ok=True)
    
    files_to_copy = ['unez_train.sh', 'training_data.h5', 'unez_train.py',
                        'config.yaml', 'all_object_model.keras', 'unez_train_supervised.py']

    for filename in files_to_copy:
        if os.path.dirname(filename):
            print(f"Skipping file in subdirectory: {filename}")
            continue

        src_file = os.path.join(os.getcwd(), filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, run_dir)
        else:
            print(f"File listed not found in base directory: {filename}")
    
    return run_dir

def run_script_in_directory(run_dir, script_path, command):
    # Change the working directory to the run directory
    os.chdir(run_dir)
    
    # Run the script with sbatch
    command = f"{command}"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    run_txt_path = os.path.join(os.getcwd(), 'config.yaml')
    if os.path.exists(run_txt_path):
        shutil.copy(run_txt_path, 'training/')
    
    os.chdir("training/")
    
    # Parent directory for all runs
    parent_dir = os.path.join(os.getcwd(), "runs")
    
    # Name for the current run 
    args = parse_arguments()
    run_name = args.run_name
    
    # Path to the main script that you want to run
    script_path = os.path.join(os.getcwd(), "scratch_paper.sh")
    
    # Create the run directory and copy the folder structure
    run_dir = create_run_directory(parent_dir, run_name)
    
    # Copy the run.txt file to the new run directory if it exists
    run_txt_path = os.path.join(os.getcwd(), 'run.txt')
    if os.path.exists(run_txt_path):
        shutil.copy(run_txt_path, run_dir)
    
    # Run the main script in the new run directory
    command = "sbatch unez_train.sh"
    run_script_in_directory(run_dir, script_path, command)
    
    # Copy the updated run.txt file back to the original directory if it has changed
    updated_run_txt_path = os.path.join(run_dir, 'run.txt')
    if os.path.exists(updated_run_txt_path) and not os.path.samefile(updated_run_txt_path, run_txt_path):
        shutil.copy(updated_run_txt_path, run_txt_path)
