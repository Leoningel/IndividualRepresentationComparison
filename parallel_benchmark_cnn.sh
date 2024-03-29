#!/bin/bash
#SBATCH --job-name=rep_comp_hpo         # Job name
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=8gb                       # Job memory request
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/parallel_%j.log   # Standard output and error log

#SBATCH --array=0-89%5                  # iterate values between 0 and 59, inclusive

pyenv local 3.11.1
source .venv/bin/activate

python run_example.py -s $(expr $SLURM_ARRAY_TASK_ID % 30) -r $(expr $SLURM_ARRAY_TASK_ID / 30) -e cnn
