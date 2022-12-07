from __future__ import annotations
import csv

import pandas as pd
import numpy as np

import utils.global_vars as gv

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import dsge_representation
from geneticengine.core.representations.grammatical_evolution.ge import ge_representation
from geneticengine.core.representations.tree.initialization_methods import Random_Production
from geneticengine.core.representations.tree.treebased import treebased_representation


def run_experiments(
        grammar,
        ff,
        ff_test,
        folder_name,
        seed,
        representation="treebased",
    ):
    novelty = 0
    if representation == "ge":
        repr = ge_representation
        repr.mutation_method = 'per_codon_mutate'
        repr.codon_prob = 0.05
    elif representation == "dsge":
        repr = dsge_representation
        repr.mutation_method = 'per_codon_mutate'
        repr.codon_prob = 0.05
    else:
        repr = treebased_representation
        novelty = int(gv.POPULATION_SIZE * 0.05)
    repr.method = Random_Production()

    def evolve(
        seed,
        mode,
    ):
        print(grammar)

        so_problem=SingleObjectiveProblem(
                    minimize=True,
                    fitness_function=ff,
                    target_fitness=0,
                )

        alg = GP(
            grammar,
            representation=repr,
            problem=so_problem,
            probability_crossover=gv.PROB_CROSSOVER,
            probability_mutation=gv.PROB_MUTATION,
            cross_over_return_one_individual=True,
            number_of_generations=gv.NUMBER_OF_ITERATIONS,
            min_init_depth=gv.MIN_INIT_DEPTH,
            min_depth=gv.MIN_DEPTH,
            max_init_depth=gv.MAX_INIT_DEPTH,
            max_depth=gv.MAX_DEPTH,
            population_size=gv.POPULATION_SIZE,
            selection_method=("tournament",gv.TOURNAMENT),
            n_elites=gv.ELITISM,
            n_novelties=novelty,
            save_to_csv=CSVCallback(
                filename=f"{gv.RESULTS_FOLDER}/{folder_name}/{representation}/{seed}.csv",
                test_data=ff_test,
                save_productions=True,
                ),
            seed=seed,
            timer_stop_criteria=mode,
        )
        (b, bf, bp) = alg.evolve(verbose=1)
        return b, bf, bp, b.count_prods(representation.genotype_to_phenotype, grammar)

    individual, fitness, phenotype, prods = evolve(seed, False)
    test_fitness = ff_test(phenotype)
    fitness = ff(phenotype)
    print(phenotype)
    print(f"With fitness: {fitness}")
    print(f"With test fitness: {test_fitness}")
    csv_row = [ fitness, test_fitness, seed, folder_name, individual.genotype, phenotype, prods ]
    with open(f"{gv.RESULTS_FOLDER}/{folder_name}/{representation}/main.csv", "a", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
    
