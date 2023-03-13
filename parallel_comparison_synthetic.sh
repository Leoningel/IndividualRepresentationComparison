#!/bin/bash
#SBATCH --nodes=1                       # Run all processes on a single node	
#SBATCH --ntasks=1                      # Run a single task		
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2000mb                     # Job memory request
#SBATCH --time=3:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/parallel_%j.log   # Standard output and error log


cd ..
pyenv local pypy3
source venv/bin/activate
pypy3 run_synthetic_example.py $@