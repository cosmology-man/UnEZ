#!/bin/bash
#SBATCH --job-name=scratch_paper
#SBATCH --partition=john7
#SBATCH --time=00:05:00  # Adjust based on expected runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G  # Adjust based on the expected memory usage
#SBATCH --output=sbatch_output/noise_gen-%J.out
#SBATCH --error=sbatch_output/noise_gen-%J.err
#SBATCH --mail-type=END,FAIL  # Get email on completion and failure
#SBATCH --mail-user=aabd@swin.edu.au

# Load Python environment
module load gcc/11.3.0
module load openmpi/4.1.4
module load python/3.10.4
module load numpy/1.22.3-scipy-bundle-2022.05
module load matplotlib/3.5.2
module load tensorflow/2.11.0-cuda-11.7.0
module load astropy/5.1.1
module load cuda/11.7.0

# Activate virtual environment
. /fred/oz149/aabd/venv/desom/bin/activate

# Run your script
python find_specobjid.py