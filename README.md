# Genetic Engine Evaluation
## Setup

1 - Run the setup file: ``./setup.sh``.

## Steps to run the comparison with PonyGE2

1 - Run ``./update.sh`` to ensure you are running the latest version in master for each framework.

2 - Execute the run evaluation file: ``./run_ponyge_comparison.sh --mode=timer``. 

You can specify a single example to be evaluated with the command ``./run_ponyge_comparison.sh example``.

You **are required** to choose what mode you want to evaluate: either by generations or timer. by using the following command:
``./run_ponyge_comparison.sh --mode=generations example`` or ``./run_ponyge_comparison.sh --mode=timer example`` 

3 - Execute the plot generator file by running: ``./ponyge_plot_generator.sh``. 

You can specify a single example to be evaluated with the command ``./ponyge_plot_generator.sh example``.
