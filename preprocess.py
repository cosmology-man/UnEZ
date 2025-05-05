import os
import shutil
import sys
import subprocess
from datetime import datetime
import argparse

if __name__ == "__main__":    
    # Copy the config file to the preprocessing directory if it exists
    run_txt_path = os.path.join(os.getcwd(), 'config.yaml')
    if os.path.exists(run_txt_path):
        shutil.copy(run_txt_path, 'preprocessing/')
    
    # Run the main script in the new run directory
    os.chdir('preprocessing')
    command = "sbatch unez_preprocess.sh"
    subprocess.run(command, shell=True, check=True)
