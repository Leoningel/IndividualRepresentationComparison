import os
import multiprocessing as mp
import sys
import src.helper as helper

# Configuration variables
GENETICENGINE_PATH = 'GeneticEngine/'
RESULTS_PATH = 'treebased_ge_comparison'

def execute_plot(plot_method, file_run_names, run_names, result_name='results/images/medians.pdf', aggregation='mean', metric='fitness', single_value=False):

    # Check the evolution time
    plot_method(file_run_names, run_names, result_name, metric, aggregation, single_value)
    

# Function to evaluate the GeneticEngine
def execute_plots(example, aggregation, file_addition=''):

    os.chdir('GeneticEngine/')
    representations = ['treebased_representation','grammatical_evolution']
    folders = [ RESULTS_PATH + '/' + example + '/' + r for r in representations ]
    output_folder = './results/images/treebased_ge_comparison/'
    helper.create_folder(output_folder)
    
    filepath = GENETICENGINE_PATH + 'geneticengine/visualization/plot_comparison.py'
    
    plot_method = helper.get_eval_method(filepath, 'plot_comparison') 
    
    print(f"Example: {example}.")
    process = mp.Process(target=execute_plot,
                            args=(plot_method,
                                  folders,
                                  representations,
                                  output_folder + f'{example}{file_addition}.pdf',
                                  aggregation
                                  ))
    process.start()
    process.join()
    # helper.copy_folder(output_folder,f'results/treebased_ge_comparison/images/')

if __name__ == '__main__':
    if len(sys.argv) < 2 or '--agg=' not in sys.argv[1]:
        raise Exception('The --agg=mean, --agg=max, --agg=min should be included after ./plot_treebased_ge_comparison.sh')
    agg = sys.argv[1].split('=')[1]
    
    examples = sys.argv[2:]
    
    for ex in examples:
        print(f"Plotting {ex}")
        execute_plots(ex, agg)
    

    

