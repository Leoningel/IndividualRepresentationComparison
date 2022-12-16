# Genetic Engine Evaluation
## Setup

1 - Run the setup file: ``./setup.sh``.

## Steps to run an example

1 - Run ``./update.sh`` to ensure you are running the latest version in master for each framework.

2 - Go into the src folder: ``cd src``

3 - Make sure the folder structure of the example you want to run is in place. You can find the folder name in the wrapper function ``run_experiment`` of your example. If it is not in place (it must have a folder in the results folder with a folder for all three representations, and a main.csv in each of these subfolders with the relevant column names), you can run the initialize.py programme: ``python initialize.py <folder_name>``.

4 - Run the programme once: ``python run_example.py -e <example_name> -s <seed (int, optional)> -r <representation (int, optional)>``, or run it for each representation with 30 seeds: ``parallel_comparison.sh -e <example_name>``.



