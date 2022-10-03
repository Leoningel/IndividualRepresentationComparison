import logging
import multiprocessing as mp
from time import perf_counter_ns

from PonyGE2.src.ponyge import evolve, preprocess, reset_ponyge

# Configuration variables
PONYGE_PATH = 'PonyGE2/'

ponyge_examples = {
    # Examples
    'pymax': 'parameters/pymax.txt',
    'vectorial': None, # TODO: Not implemented
    'regression': 'parameters/regression.txt',
    'santafe': None, # TODO: Not implemented
    'string_match': 'parameters/string_mat.txt',
    'seed_run_target': 'parameters/seed_run_target.txt',
    'GE_parse': 'parameters/GE_parse.txt',
    # Progsys
    'number_io': None,
    'median': None,
    'smallest': None,
    'sum_of_squares': None,
}


def execute_evaluation(params, queue):

    # Check the processing time
    processing_time = perf_counter_ns()
    preprocess(params)
    processing_time = perf_counter_ns() - processing_time

    # Check the evolution time
    evolution_time = perf_counter_ns()
    evolve()
    evolution_time = perf_counter_ns() - evolution_time

    reset_ponyge()

    queue.put(processing_time)
    queue.put(evolution_time)


# Function to evaluate PonyGE
def evaluate_ponyge(examples):
        
    if len(examples) > 0:
        run_examples = dict([(name, function) for name, function in ponyge_examples.items() if name in examples])

    else:
        run_examples = ponyge_examples

    for name, parameter_path in run_examples.items():        
        
        logging.info(f"PonyGE: Executing the example: {name}")

        # Collect the path
        filepath = PONYGE_PATH + parameter_path

        # Accumulate the results
        result_process_time = list() 
        result_evolution_time = list()

        # Run 30 times with 30 different seeds
        for seed in range(1):
            queue = mp.Queue()

            parameters = ['--parameters', parameter_path, '--random_seed', str(seed)]

            process = mp.Process(target=execute_evaluation, 
                                    args=(parameters, queue))
            process.start()
            process.join()

            result_process_time.append(queue.get())
            result_evolution_time.append(queue.get())
        