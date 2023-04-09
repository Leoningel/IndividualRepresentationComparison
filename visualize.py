from geneticengine.visualization.per_gen_comparisons import (
    plot_fitness_comparison,
    plot_nodes_comparison,
    plot_test_fitness_comparison,
)

# python visualize.py patient1 RandomProd/try3
if __name__ == "__main__":
    import sys

    try:
        folder = sys.argv[1]
    except Exception:
        print("You must specify a folder!")
        sys.exit(-1)

    folder_names = [
        f"results/{folder}/dsge",
        f"results/{folder}/treebased",
        f"results/{folder}/ge",
    ]
    labels = ["dSGE", "Tree-based", "GE"]
    folders = ["dsge", "treebased", "ge"]

    plot_fitness_comparison(
        folder_names=folder_names,
        labels=labels,
        labels_name="Representation",
        file_name=f"results/{folder}/fitness_comp_.pdf",
        title=f"Fitness comparison ({folder})",
    )
    plot_nodes_comparison(
        folder_names=folder_names,
        labels=labels,
        labels_name="Representation",
        file_name=f"results/{folder}/nodes_comp_.pdf",
        title=f"Nodes comparison ({folder})",
    )
    plot_nodes_comparison(
        y_axis="Depth",
        folder_names=folder_names,
        labels=labels,
        labels_name="Representation",
        file_name=f"results/{folder}/depth_comp_.pdf",
        title=f"Depth comparison ({folder})",
    )
    try:
        plot_test_fitness_comparison(
            folder_names=folder_names,
            labels=labels,
            labels_name="Representation",
            file_name=f"results/{folder}/test_fitness_comp_.pdf",
            title=f"Test fitness comparison ({folder})",
        )
    except Exception:
        pass
    # for idx, folder_name in enumerate(folder_names):
    #     plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, )", file_name=f'results/{folder}/{folders[idx]}/productions_comp_.pdf', keep_in_prods=[ 'VarCh', 'VarGl', 'VarIns' ])
    #     plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, )", file_name=f'results/{folder}/{folders[idx]}/glucose_productions_comp_.pdf', keep_in_prods=[ 'VarGl2', 'VarGl4', 'VarGl7' ])
    #     plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, )", file_name=f'results/{folder}/{folders[idx]}/carbsh_productions_comp_.pdf', keep_in_prods=[ 'VarCh1011', 'VarCh1213', 'VarCh1415', 'VarCh1617', 'VarCh1819', 'VarCh2021', 'VarCh2223' ])
    #     plot_prods_comparison(folder_name, title=f"Production comparison ({labels[idx]}, )", file_name=f'results/{folder}/{folders[idx]}/insuline_productions_comp_.pdf', keep_in_prods=[ 'VarIns24', 'VarIns2526', 'VarIns2728', 'VarIns2930', 'VarIns3132', 'VarIns3334', 'VarIns3536', 'VarIns3738' ])
    #     pass
