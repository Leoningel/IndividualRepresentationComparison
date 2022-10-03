from lib2to3.pgen2.pgen import DFAState
import sys
import src.helper as helper

import itertools
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def draw_barplot_and_boxplot(df: pd.DataFrame, outbasename: str, what_to_plot: str = "Fitness"):
    """
    Draws a barplot that compares the two tools
    relatively to some metric passed in the 'what_to_plot'
    argument, for all the existing examples.
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({"font.family": "serif"})
    
    df["Relative {}".format(what_to_plot)] = df.apply(
        lambda x: x[what_to_plot]
        / df[
            df.Tool.str.contains("PonyGE2") & df.Benchmark.str.contains(x.Benchmark)
        ].mean(),
        axis=1,
    )
    
    # Adjust relative fitness for maximization problems 
    # so that the comparison is the same for all exampes
    if what_to_plot == "Fitness":
        
        for index, row in df.iterrows():
            if  (not (row['Benchmark'] == 'game_of_life' or row['Benchmark'] == 'classification')) and row['Tool'] != 'PonyGE2':                
                df.loc[index, 'Relative {}'.format(what_to_plot)] = 1 / df.loc[index, 'Relative {}'.format(what_to_plot)]

    #palette = {"PonyGE2": "steelblue", "GEngine": "pink"}

    for _kind in ['bar', 'box']:
        ax = sns.catplot(
            kind=_kind,
            data=df,
            x="Benchmark",
            y="Relative {}".format(what_to_plot),
            #palette=palette,
            hue="Tool",
        )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/{}_{}.pdf".format(outbasename, _kind))
        plt.close()


def draw_violin_with_facets(
    df: pd.DataFrame, outbasename: str, what_to_plot: str = "Fitness"
):
    """
    Draws violin plots for all examples.

    Each facet represents an example, which
    allows for the scale of the 'what_to_plot' to reflect
    absolute values, rather than relative.
    """
    sns.set_style({"font.family": "serif"})
    sns.set(font_scale=0.75) 
    
    to_replace = {
        "classification": "Classification",
        "game_of_life": "Game of life",
        "regression": "Regression",
        "string_match": "String match",
        "vectorialgp": "VectorialGP",
    }

    df = df.replace(to_replace)
    if what_to_plot == "Fitness":
        maximize_examples = ['Classification', 'Game of life']
        df['Fitness'] = df.apply(lambda x: x.Fitness if x.Benchmark in maximize_examples else x.Fitness * (-1), axis=1)

    elif what_to_plot == "Time":
        df['Time (s)'] = df.Time
        what_to_plot = "Time (s)"    
    
    #palette = {"PonyGE2": "darkgrey", "GEngine": "indianred"}
    palette = {"PonyGE2": "steelblue", "GEngine": "pink"}
  
    g = sns.catplot(x="Tool",
                    y=what_to_plot,
                    sharey=False,
                    sharex=False,
                    palette=palette,
                    height=1.7,
                    aspect=1, 
                    kind="violin",
                    col="Benchmark",
                    col_wrap=5,
                    cut=0,
                    fmt='.2',
                    data=df)

    g.set_axis_labels("", what_to_plot).set_titles("{col_name}").despine(left=True)
    #g.set_xticklabels(fontsize=8).set_yticklabels(fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/{}.pdf".format(outbasename))
    sns.set(font_scale=1) 

if __name__ == "__main__":

    examples = sys.argv[1:]

    # Obtain the generation files to make the plots
    ponyge_gens = helper.import_data(examples, "results/ponyge/", "generations", ",")
    gengy_gens = helper.import_data(examples, "results/gengine/", "generations", ",")

    # Obtain the timer files to make the plots
    ponyge_timer = helper.import_data(examples, "results/ponyge/", "timer", ",")
    gengy_timer = helper.import_data(examples, "results/gengine/", "timer", ",")
    helper.create_folder("plots/")

    plot_info = {
        "title": "",
        "mode": "",
        "column": "",
        "example": "",
    }

    # #########################################################################
    # Create the total performance column
    if len(ponyge_gens) == len(gengy_gens) != 0:
        print("Running time per generation plot")
        for example in examples:
            if not example in ponyge_gens or not example in gengy_gens:
                continue

            ponyge_df = ponyge_gens[example]
            gengine_df = gengy_gens[example]

            cols = ["processing_time", "evolution_time"]

            # Convert from ns to s
            for df, column in itertools.product([ponyge_df, gengine_df], cols):
                df[column] = df[column].apply(lambda x: x * pow(10, -9))

            ponyge_df["total"] = (
                ponyge_df["processing_time"] + ponyge_df["evolution_time"]
            )
            gengine_df["total"] = (
                gengine_df["processing_time"] + gengine_df["evolution_time"]
            )

        # #########################################################################
        # Create the Merged Plot of time taken to do the evolution
        cols = ["Tool", "Benchmark", "Time"]
        rows = []

        for column, example in enumerate(examples):
            if not example in ponyge_gens or not example in gengy_gens:
                continue

            ponyge_df = ponyge_gens[example]
            gengine_df = gengy_gens[example]

            ponyge_rows = list(ponyge_df["total"].values)
            gengine_rows = list(gengine_df["total"].values)

            for val in ponyge_rows:
                rows.append(["PonyGE2", example, val])

            for val in gengine_rows:
                rows.append(["GEngine", example, val])

        merged_dataframe = pd.DataFrame(data=rows, columns=cols)
        draw_barplot_and_boxplot(
            merged_dataframe, outbasename="merged_plots_time", what_to_plot="Time"
        )
        draw_violin_with_facets(merged_dataframe, outbasename="merged_plots_time_faceted", what_to_plot="Time")

    # #########################################################################
    # Create the Merged Plot of Fitness in timer mode
    if len(ponyge_timer) == len(gengy_timer) != 0:
        print("Running fitness within time limit plot")

        cols = ["Tool", "Benchmark", "Fitness"]
        rows = []

        for column, example in enumerate(examples):
            if not example in ponyge_timer or not example in gengy_timer:
                continue

            ponyge_df = ponyge_timer[example]
            gengine_df = gengy_timer[example]

            ponyge_rows = list(ponyge_df["best_fitness"].values)
            gengine_rows = list(gengine_df["best_fitness"].values)

            for val in ponyge_rows:
                rows.append(["PonyGE2", example, val])

            for val in gengine_rows:
                rows.append(["GEngine", example, val])

        merged_dataframe = pd.DataFrame(data=rows, columns=cols)

        draw_barplot_and_boxplot(
            merged_dataframe, outbasename="merged_plots_fitness", what_to_plot="Fitness"
        )
        
        draw_violin_with_facets(merged_dataframe, outbasename="merged_plots_fitness_faceted", what_to_plot="Fitness")

       
