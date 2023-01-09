
import csv
import os
import sys

import examples.utils.global_vars as gv

RESULTS_FOLDER = gv.RESULTS_FOLDER 

if __name__ == "__main__":
    representations = [ "treebased", "ge", "dsge" ]

    folder_name = sys.argv[1]
    columns = [ "fitness", "test_fitness", "seed", "benchmark_name", "genotype", "phenotype", "prods", "depth", "nodes" ]

    try:
        os.mkdir(f"{RESULTS_FOLDER}")
    except:
        pass

    synthetic = False
    if sys.argv[1] == '-s':
            synthetic = True

    if synthetic:
        folder_name = "synthetic"
        columns = [ "fitness", "seed", "benchmark_name", "genotype", "phenotype", "prods", "fitness_function_level", "target_ind", "MAX_DEPTH", "MAX_INIT_DEPTH", "POPULATION_SIZE", "ELITSM", "TARGET_FITNESS", "PROBABILITY_CO", "PROBABILITY_MUT", "NOVELTY", "TOURNAMENT_SIZE", "grammar_depth_min", "grammar_depth_max", "grammar_n_non_terminals", "grammar_n_prods_occurrences", "grammar_n_recursive_prods", "depth", "nodes" ]
        
    os.mkdir(f"{RESULTS_FOLDER}/{folder_name}")
    for representation in representations:
        os.mkdir(f"{RESULTS_FOLDER}/{folder_name}/{representation}")
        with open(f"{RESULTS_FOLDER}/{folder_name}/{representation}/main.csv", "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)