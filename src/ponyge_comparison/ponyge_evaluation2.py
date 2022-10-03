import subprocess
import logging
import os 
import shutil

import src.helper as helper

# Configuration variables
PONYGE_PATH = 'PonyGE2/'

ponyge_examples = {
    # Examples
    # 'pymax': 'parameters/pymax.txt',
    'game_of_life': 'parameters/game_of_life.txt',
    'regression': 'parameters/regression.txt',
    'classification': 'parameters/classification.txt',
    'string_match': 'parameters/string_match.txt',
    'vectorialgp': 'parameters/vectorialgp.txt',
    
    # Progsys
    # 'number_io': 'parameters/number_io.txt',
    # 'smallest': 'parameters/smallest.txt',
    # 'median': 'parameters/median.txt',
    # 'sum_of_squares': 'parameters/sum_of_squares.txt',
    # 'vector_average': 'parameters/vector_average.txt',
}

def execute_tests(name, parameter_path, mode):

    search_mode = 'search_loop_with_timer' if mode == 'timer' else 'search_loop'

    # Collect the path
    filepath = PONYGE_PATH + 'src/ponyge_eval.py'
    parameter_path = PONYGE_PATH + parameter_path


    # Run 30 times with 30 different seeds
    for seed in range(30):
        subprocess.call(["python", filepath, 
                            '--parameters', parameter_path, 
                            '--random_seed', str(seed),
                            '--search_loop', search_mode])
    
    shutil.rmtree(f'results/ponyge/{name}')

# Function to evaluate PonyGE
def evaluate_ponyge2(examples, mode):
    for e in examples:
        assert e in ponyge_examples.keys(), "Example '{} is not valid.\nList of available example names:\n{}".format(e, '\n'.join(list(ponyge_examples.keys())))
    
    if len(examples) > 0:
        run_examples = dict([(name, function) for name, function in ponyge_examples.items() if name in examples and function != None])

    else:
        run_examples = ponyge_examples

    helper.create_folder('results/ponyge/')

    for name, parameter_path in run_examples.items():        
        
        logging.info(f"PonyGE: Executing the example: {name}")
    
        # Write the header of the times file
        f = open(f"results/ponyge/{name}_{mode}.csv", "w")
    
        if mode == 'generations':
            f.write("processing_time,evolution_time")
        if mode == 'timer':
            f.write("best_fitness")
        
        f.close()

        execute_tests(name, parameter_path, mode)