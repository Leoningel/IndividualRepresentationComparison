from argparse import ArgumentParser
import csv
import os
import shutil
import src.helper as helper
from datetime import datetime

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

results_path = 'results'

modes = [ "generations", "time" ]
example_names = list(examples.keys())
representations = [ "treebased_representation", "ge", "dsge" ]

def clean_results(archive_name):
    shutil.move(results_path, f"./archive/{archive_name}")
    os.mkdir(results_path)
    for mode in modes:
        helper.create_folder(f'{results_path}/{mode}')
        for example_name in example_names:
            helper.create_folder(f'{results_path}/{mode}/{example_name}')
            for representation in representations:
                helper.create_folder(f'{results_path}/{mode}/{example_name}/{representation}')
                with open(f"{results_path}/{mode}/{example_name}/{representation}/main.csv", "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([ "mode", "example", "representation", "seed", "train_score", "time", "best_ind" ])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', "--archive_name", dest='archive_name', type=str, default=str(datetime.now().timestamp()))
    args = parser.parse_args()
    clean_results(args.archive_name)
    

        
