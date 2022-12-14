#!/bin/bash
#SBATCH --job-name=rep_comp_cnn         # Job name
#SBATCH --nodes=1                       # Run all processes on a single node	
#SBATCH --ntasks=1                      # Run a single task		
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=10gb                      # Job memory request
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/parallel_%j.log   # Standard output and error log

#SBATCH --array=0-89%5                  # iterate values between 0 and 59, inclusive

cd ..
pip install -r requirements.txt
cd Gengy_implementation
python examples/CNN.py -s $(expr $SLURM_ARRAY_TASK_ID % 30) -r $(expr $SLURM_ARRAY_TASK_ID / 30)