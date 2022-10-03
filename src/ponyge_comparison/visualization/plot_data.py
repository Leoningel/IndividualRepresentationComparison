import sys
import src.helper as helper

import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns

def plot_df(ponyge_df, gengine_df, plot_info):
    
    ponyge_vals = ponyge_df[plot_info['column']].values
    gengy_vals = gengine_df[plot_info['column']].values
  
    ax = sns.violinplot(data=[ponyge_vals, gengy_vals])
    #ax = sns.swarmplot(data=[ponyge_vals, gengy_vals], color=".25")
    
    tix = ['PonyGE', 'GeneticEngine']
    
    plt.xticks([0,1], tix, rotation=0, fontsize=12)
    plt.title(plot_info['example'])
    
    if plot_info['mode'] == 'generations':
        ax.set_ylabel("Time (s)", fontsize=12)
    else:
        ax.set_ylabel("Fitness (mae)", fontsize=12)
    
    ax.set_xlabel('Tools', fontsize=12)

    ax.set_ylim(ymin=0)
    
    plt.savefig(f"plots/{plot_info['example']}_{plot_info['column']}_{plot_info['mode']}.pdf")
    plt.close()


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    examples = sys.argv[1:]

    # Obtain the generation files to make the plots
    ponyge_gens = helper.import_data(examples, 'results/ponyge/', 'generations', ',')
    gengy_gens = helper.import_data(examples, 'results/gengine/', 'generations', ',')

    # Obtain the timer files to make the plots
    ponyge_timer = helper.import_data(examples, 'results/ponyge/', 'timer', ',')
    gengy_timer = helper.import_data(examples, 'results/gengine/', 'timer', ',')
    helper.create_folder("plots/")

    plot_info = {'title': '',
        'mode': '',
        'column': '',
        'example': '',
    }

    # #########################################################################
    # Generate the gens plot files
    for example in examples:

        # Update the plot informations    
        plot_info['mode'] = 'generations'
        plot_info['example'] = example 

        if not example in ponyge_gens or not example in gengy_gens:
            continue
        
        ponyge_df = ponyge_gens[example]
        gengine_df = gengy_gens[example]

        cols = ['processing_time', 'evolution_time']

        # Convert from ns to s
        for df, column in itertools.product([ponyge_df, gengine_df], cols):
            df[column] = df[column].apply(lambda x: x * pow(10, -9))
        
        ponyge_df['total'] = ponyge_df['processing_time'] + ponyge_df['evolution_time']
        gengine_df['total'] = gengine_df['processing_time'] + gengine_df['evolution_time']

        # Plot the Processing Time
        plot_info['title'] = 'Processing Time per Tool'
        plot_info['column'] = 'processing_time'
        plot_df(ponyge_df, gengine_df, plot_info)

        # Plot the Evolution time
        plot_info['title'] = 'Evolution Time per Tool'
        plot_info['column'] = 'evolution_time'
        plot_df(ponyge_df, gengine_df, plot_info)
        
        # Plot the Total time
        plot_info['title'] = 'Total Time per Tool'
        plot_info['column'] = 'total'
        plot_df(ponyge_df, gengine_df, plot_info)


    # #########################################################################
    # Generate the timer plot files
    for example in examples:

        # Update the plot informations    
        plot_info['mode'] = 'timer'
        plot_info['example'] = example 
        plot_info['title'] = 'Best fitness per Tool after time threshold'
        plot_info['column'] = 'best_fitness'

        if not example in ponyge_timer or not example in gengy_timer:
            continue
        
        ponyge_df = ponyge_timer[example]
        gengine_df = gengy_timer[example]

        plot_df(ponyge_df, gengine_df, plot_info)
        