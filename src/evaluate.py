import csv
import pandas as pd
import sys
import configs.standard as gv



def evaluate_single_patient(folder_name):
    main_file = pd.read_csv(f"{folder_name}/main.csv")
    fitnesses = main_file["fitness"]
    mean_fitness = fitnesses.mean()
    std_fitness = fitnesses.std()
    
    test_fitnesses = main_file["test_fitness"]
    mean_test_fitness = test_fitnesses.mean()
    std_test_fitness = test_fitnesses.std()

    return (mean_fitness, std_fitness), (mean_test_fitness, std_test_fitness)

def evaluate_all_patients(folder_name):
    with open(f"{folder_name}/means_and_stds.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([ "patient", "mean_fitness", "std_fitness", "mean_test_fitness", "std_test_fitness" ])
    
    for i in range(10):
        (mf, sf), (mtf, stf) = evaluate_single_patient(f"{folder_name}/patient{i + 1}")
        csv_row = [ f"patient{i + 1}", mf, sf, mtf, stf ]
        with open(f"{folder_name}/means_and_stds.csv", "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(csv_row)

# python evaluate.py RandomProd/100_mut/dsge
if __name__ == "__main__":
    folder_name = sys.argv[1]
    
    evaluate_all_patients(f"{gv.RESULTS_FOLDER}/{folder_name}")
