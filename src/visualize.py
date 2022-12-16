from geneticengine.visualization.per_gen_comparisons import plot_fitness_comparison, plot_nodes_comparison, plot_prods_comparison, plot_test_fitness_comparison

import configs.standard as gv

# python visualize.py patient1 RandomProd/try3
if __name__ == "__main__":
    import sys
    try:
        patient = sys.argv[1]
    except:
        print("You must specify a patient!")
    try:
        folder = sys.argv[2]
    except:
        print("You must specify a folder!")

    folder_names = [f'{gv.RESULTS_FOLDER}/{folder}/dsge/{patient}', f'{gv.RESULTS_FOLDER}/{folder}/treebased/{patient}', f'{gv.RESULTS_FOLDER}/{folder}/ge/{patient}']
    labels = [ 'dSGE', 'Tree-based', 'GE' ]
    folders = [ 'dsge', 'treebased', 'ge' ]

    plot_fitness_comparison(folder_names=folder_names, labels=labels, labels_name='Representation',file_name=f'{gv.RESULTS_FOLDER}/{folder}/fitness_comp_{patient}.pdf', title=f'Fitness comparison ({patient})')
    plot_nodes_comparison(folder_names=folder_names, labels=labels, labels_name='Representation',file_name=f'{gv.RESULTS_FOLDER}/{folder}/nodes_comp_{patient}.pdf', title=f'Nodes comparison ({patient})')
    plot_nodes_comparison(y_axis='Depth', folder_names=folder_names, labels=labels, labels_name='Representation',file_name=f'{gv.RESULTS_FOLDER}/{folder}/depth_comp_{patient}.pdf', title=f'Depth comparison ({patient})')
    plot_test_fitness_comparison(folder_names=folder_names, labels=labels, labels_name='Representation',file_name=f'{gv.RESULTS_FOLDER}/{folder}/test_fitness_comp_{patient}.pdf', title=f'Test fitness comparison ({patient})')
    for idx, folder_name in enumerate(folder_names):
        plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, {patient})", file_name=f'{gv.RESULTS_FOLDER}/{folder}/{folders[idx]}/productions_comp_{patient}.pdf', keep_in_prods=[ 'VarCh', 'VarGl', 'VarIns' ])
        plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, {patient})", file_name=f'{gv.RESULTS_FOLDER}/{folder}/{folders[idx]}/glucose_productions_comp_{patient}.pdf', keep_in_prods=[ 'VarGl2', 'VarGl4', 'VarGl7' ])
        plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, {patient})", file_name=f'{gv.RESULTS_FOLDER}/{folder}/{folders[idx]}/carbsh_productions_comp_{patient}.pdf', keep_in_prods=[ 'VarCh1011', 'VarCh1213', 'VarCh1415', 'VarCh1617', 'VarCh1819', 'VarCh2021', 'VarCh2223' ])
        plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, {patient})", file_name=f'{gv.RESULTS_FOLDER}/{folder}/{folders[idx]}/insuline_productions_comp_{patient}.pdf', keep_in_prods=[ 'VarIns24', 'VarIns2526', 'VarIns2728', 'VarIns2930', 'VarIns3132', 'VarIns3334', 'VarIns3536', 'VarIns3738' ])
        pass