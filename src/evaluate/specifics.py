from argparse import ArgumentParser
from typing import List
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt

to_replace = {
    "treebased_representation"  : "CFG-GP",
    "ge"                        : "GE",
    "dsge"                      : "DSGE",
}
representations = [ "dsge", "ge", "treebased_representation" ]

def visualise_compare_reps(folder: str, column: str = 'fitness', per_column: str = 'number_of_the_generation', fitness_name: str = 'MSE'):

    li = []

    for rep in representations:
        all_files = glob.glob(f"{folder}/{rep}/*.csv")

        for idx, filename in enumerate(all_files):
            if "main" not in filename:
                print(f"{round((idx/len(all_files)) * 100,1)} %", end='\r')
                df = pd.read_csv(filename, index_col=None, header=0)
                df = df[[column, per_column]]
                df["representation"] = rep
                df = df.replace(to_replace)
                li.append(df)
        
    df = pd.concat(li, axis=0, ignore_index=True)
    
    plt.close()
    sns.set_style({"font.family": "serif"})
    sns.set(font_scale=0.75)

    new_column = column 
    if column != 'nodes':
        if new_column == 'test_fitness':
            new_column = 'test fitness'
        new_column = f'{column} ({fitness_name})'
    new_per_column = 'Generations'
    df[new_column] = df[[column]]
    df[new_per_column] = df[[per_column]]

    a = sns.lineplot(
            data=df,
            x = new_per_column,
            y = new_column,
            hue = 'representation'
            )

    a.set_title(f"{new_column} comparison")
    path = f"{folder}/00_{new_column}.pdf"
    plt.savefig(path)
    print(f"Saved figure to {path}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', "--folder_name", dest='folder_name', type=str)
    args = parser.parse_args()
    visualise_compare_reps(args.folder_name, fitness_name='santafe - maximize')
    

    
