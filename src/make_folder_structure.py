from argparse import ArgumentParser
import csv
import os
import shutil
import src.helper as helper
from datetime import datetime
import global_vars as gv


modes = [ "generations", "time" ]
example_names = list(gv.examples.keys())
representations = [ "treebased_representation", "ge", "dsge" ]

def clean_results(archive_name):
    shutil.move(gv.RESULTS_PATH, f"./archive/{archive_name}")
    os.mkdir(gv.RESULTS_PATH)
    for mode in modes:
        helper.create_folder(f'{gv.RESULTS_PATH}/{mode}')
        for example_name in example_names:
            helper.create_folder(f'{gv.RESULTS_PATH}/{mode}/{example_name}')
            for representation in representations:
                helper.create_folder(f'{gv.RESULTS_PATH}/{mode}/{example_name}/{representation}')
                with open(f"{gv.RESULTS_PATH}/{mode}/{example_name}/{representation}/main.csv", "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([ "mode", "example", "representation", "seed", "train_score", "time", "best_ind" ])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', "--archive_name", dest='archive_name', type=str, default=str(datetime.now().timestamp()))
    args = parser.parse_args()
    clean_results(args.archive_name)
    

        
