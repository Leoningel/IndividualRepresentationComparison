#!/bin/bash

#SBATCH --job-name=synthetic         # Job name
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2000mb                     # Job memory request
#SBATCH --time=0:10:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/synthetic_%j.log   # Standard output and error log


#SBATCH --array=0-999                   # iterate values between 0 and 59, inclusive


cd ..
pyenv local 3.11.1
source venv/bin/activate
cd src
python run_synthetic_example.py --seed $SLURM_ARRAY_TASK_ID $@
