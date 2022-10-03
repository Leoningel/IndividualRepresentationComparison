import csv
from optparse import OptionParser
import os
import sys
import logging
import multiprocessing as mp
from time import perf_counter_ns, process_time
import time
import pandas as pd
import src.helper as helper

# Configuration variables
GENETICENGINE_PATH = 'GeneticEngine/'

examples = {
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

if __name__ == '__main__':
    example_names = list(examples.keys())
    representations = [ "treebased_representation", "ge", "dsge" ]
    timed = False

    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", type="int")
    parser.add_option("-e", "--example", dest="example", type="int", default=0)
    parser.add_option("-r", "--representation", dest="representation", type="int")
    (options, args) = parser.parse_args()

    seed = options.seed
    example_name = example_names[options.example]
    representation = representations[options.representation]
    print(seed, example_name, representation)

    example_path = GENETICENGINE_PATH + examples[example_name]
    evol_method = helper.get_eval_method(example_path, 'evolve')

    mode = "generations"
    if timed:
        mode = "time"
    dest_file = f"results/{mode}/{example_name}/{representation}/{seed}.csv"

    start = time.time()
    b, bf = evol_method(seed, timed, dest_file, representation)
    end = time.time()
    csv_row = [ mode, example_name, representation, seed, bf, (end - start), b ]
    with open(f"./results/{mode}/{example_name}/{representation}/main.csv", "a", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
    
    
    print(b, bf)
    
    

