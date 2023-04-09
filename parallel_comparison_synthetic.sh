#!/bin/bash
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2000mb                     # Job memory request
#SBATCH --time=0:10:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/parallel_%j.log   # Standard output and error log


cd ..
pyenv local 3.10.9
source venv/bin/activate
cd src
python run_synthetic_example.py $@
