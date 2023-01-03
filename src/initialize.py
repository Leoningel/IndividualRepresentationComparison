
import csv
import os
import sys

import examples.utils.global_vars as gv

RESULTS_FOLDER = gv.RESULTS_FOLDER 

if __name__ == "__main__":
    representations = [ "treebased", "ge", "dsge" ]

    folder_name = sys.argv[1]

    try:
        if sys.argv[2] == '-s':
            folder_name = f"synthetic/{folder_name}"
        try:
            os.mkdir(f"{RESULTS_FOLDER}/synthetic")
        except:
            pass
    except:
        pass

    try:
        os.mkdir(f"{RESULTS_FOLDER}")
    except:
        pass
        
    os.mkdir(f"{RESULTS_FOLDER}/{folder_name}")
    for representation in representations:
        os.mkdir(f"{RESULTS_FOLDER}/{folder_name}/{representation}")
        with open(f"{RESULTS_FOLDER}/{folder_name}/{representation}/main.csv", "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow([ "fitness", "test_fitness", "start_test_set", "seed", "fold", "file_name", "genotype", "phenotype", "prods" ])