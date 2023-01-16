#!/bin/bash
#SBATCH --job-name=synth                # Job name
#SBATCH --nodes=1                       # Run all processes on a single node	
#SBATCH --ntasks=1                      # Run a single task		
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=100mb                     # Job memory request
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/parallel_%j.log   # Standard output and error log

#SBATCH --array=0-1000                    # iterate values between 0 and 59, inclusive

cd ..
pip install -r requirements.txt
cd src
python run_synthetic_example.py -s $(expr $SLURM_ARRAY_TASK_ID)