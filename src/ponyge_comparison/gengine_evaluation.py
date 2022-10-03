import os
import sys
import logging
import multiprocessing as mp
from time import perf_counter_ns, process_time
import pandas as pd
import src.helper as helper

# Configuration variables
GENETICENGINE_PATH = 'GeneticEngine/'

gengine_examples = {
    # Examples
    # 'santafe': 'examples/santafe.py',
    # 'pymax': 'examples/pymax.py',
    'game_of_life': 'examples/game_of_life.py',
    'regression': 'examples/regression.py',
    'classification': 'examples/classification.py',
    'string_match': 'examples/string_match.py',
    'vectorialgp': 'examples/vectorialgp_example.py',
    
    # Progsys
    # 'number_io': 'examples/progsys/Number_IO.py',
    # 'smallest': 'examples/progsys/Smallest.py',
    # 'median': 'examples/progsys/Median.py',
    # 'sum_of_squares': 'examples/progsys/Sum_of_Squares.py',
    # 'vector_average': 'examples/progsys/Vector_Average.py',
}

def execute_evaluation(preprocess_method, evol_method, seed, mode, queue):

    # Check the processing time
    processing_time = perf_counter_ns()
    algorithm = preprocess_method()
    processing_time = perf_counter_ns() - processing_time

    # Check the evolution time
    evolution_time = perf_counter_ns()
    best_individual, best_fitness = evol_method(algorithm, seed, mode == 'timer', representation = 'treebased_representation')
    evolution_time = perf_counter_ns() - evolution_time

    if mode == 'timer':
        queue.put(best_fitness)
    else:
        queue.put(processing_time)
        queue.put(evolution_time)
    
    #queue.put(best_individual)

# Function to evaluate the GeneticEngine
def evaluate_geneticengine(examples: list, mode):

    os.chdir('GeneticEngine/')
    
    for e in examples:
        assert e in gengine_examples.keys(), "Example '{} is not valid.\nList of available example names:\n{}".format(e, '\n'.join(list(gengine_examples.keys())))
    
    dataframe = dict()

    if len(examples) > 0:
        run_examples = dict([(name, function) for name, function in gengine_examples.items() if name in examples])

    else:
        run_examples = gengine_examples

    for name, path in run_examples.items():

        logging.info(f"GEngine: Executing the example: {name}")

        # Collect the path
        filepath = GENETICENGINE_PATH + path
       
        # Obtain the preprocessing and evolution method
        preprocess_method = helper.get_eval_method(filepath, 'preprocess') 
        evol_method = helper.get_eval_method(filepath, 'evolve')
        
        # Accumulate the results
        output_list = list()
        
        # Run 30 times with 30 different seeds
        for seed in range(30):
            print("Run:", seed)
            queue = mp.Queue()

            process = mp.Process(target=execute_evaluation,
                                    args=(preprocess_method, 
                                        evol_method,
                                        seed,
                                        mode,
                                        queue))
            process.start()
            process.join()
            
            output = [queue.get()] if mode == 'timer' else [queue.get(), queue.get()]
            
            output_list.append(output)

        cols = ['best_fitness'] if mode == 'timer' else ['processing_time', 'evolution_time'] 
        
        dataframe[name] = pd.DataFrame(output_list, columns=cols)
    
    os.chdir('..')
    path = 'results/gengine/'
    helper.write_to_csv_times(dataframe, mode, path)