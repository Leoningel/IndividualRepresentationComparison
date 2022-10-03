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
import global_vars as gv


if __name__ == '__main__':
    example_names = list(gv.examples.keys())
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

    example_path = gv.GENETICENGINE_PATH + gv.examples[example_name]
    evol_method = helper.get_eval_method(example_path, 'evolve')

    mode = "generations"
    if timed:
        mode = "time"
    dest_file = f"{gv.RESULTS_PATH}/{mode}/{example_name}/{representation}/{seed}.csv"

    start = time.time()
    b, bf = evol_method(seed, timed, dest_file, representation)
    end = time.time()
    csv_row = [ mode, example_name, representation, seed, bf, (end - start), b ]
    with open(f"./{gv.RESULTS_PATH}/{mode}/{example_name}/{representation}/main.csv", "a", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
    
    
    print(b, bf)
    
    

