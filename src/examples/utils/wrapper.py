from __future__ import annotations
import csv

import pandas as pd
import numpy as np
from random import Random

import examples.utils.global_vars as gv

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import Grammar
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
        benchmark_name,
        seed,
        params: dict,
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
        novelty = int(params['POPULATION_SIZE'] * 0.05)
    repr.method = Random_Production()

    def evolve(
        seed,
        mode,
    ):
        print(grammar)

        so_problem=SingleObjectiveProblem(
                    minimize=params['MINIMIZE'],
                    fitness_function=ff,
                    target_fitness=params['TARGET_FITNESS'],
                )

        alg = GP(
            grammar,
            representation=repr,
            problem=so_problem,
            probability_crossover=gv.PROB_CROSSOVER,
            probability_mutation=gv.PROB_MUTATION,
            cross_over_return_one_individual=True,
            number_of_generations=params['NUMBER_OF_ITERATIONS'],
            min_init_depth=params['MIN_INIT_DEPTH'],
            min_depth=params['MIN_DEPTH'],
            max_init_depth=params['MAX_INIT_DEPTH'],
            max_depth=params['MAX_DEPTH'],
            population_size=params['POPULATION_SIZE'],
            selection_method=("tournament",gv.TOURNAMENT),
            n_elites=params['ELITSM'],
            n_novelties=novelty,
            save_to_csv=CSVCallback(
                filename=f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/{seed}.csv",
                test_data=ff_test,
                save_genotype_as_string=True,
                save_productions=True,
                ),
            seed=seed,
            timer_stop_criteria=mode,
        )
        (b, bf, bp) = alg.evolve(verbose=1)
        return b, bf, bp, b.count_prods(repr.genotype_to_phenotype, grammar)

    individual, fitness, phenotype, prods = evolve(seed, False)
    test_fitness = None
    if ff_test:
        test_fitness = ff_test(phenotype)
    fitness = ff(phenotype)
    print(phenotype)
    print(f"With fitness: {fitness}")
    print(f"With test fitness: {test_fitness}")
    csv_row = [ fitness, test_fitness, seed, benchmark_name, individual.genotype, phenotype, prods ]
    with open(f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/main.csv", "a", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)


def run_synthetic_experiments(
        grammar: Grammar,
        ffs_and_target_ind,
        benchmark_name,
        seed,
    ):
    random = Random(seed)
    pop_size = round(random.normalvariate(50, 5))
    
    params = {
        'MINIMIZE': True,
        'NUMBER_OF_ITERATIONS': 25,
        'MAX_INIT_DEPTH': round(random.normalvariate(5, 1.5)), # vary
        'POPULATION_SIZE': pop_size, # vary
        'ELITSM': round(min(max(random.normalvariate(5, 2), 0), pop_size/10)), # vary between 0 - 10%
        'TARGET_FITNESS': 0,
        'PROBABILITY_CO': min(1, max(0, random.normalvariate(0.6, .3))),
        'PROBABILITY_MUT': min(1, max(0, random.normalvariate(0.6, .3))),
        'NOVELTY': round(min(max(random.normalvariate(5, 2), 0), pop_size/10)), # vary between 0 - 10%
        'TOURNAMENT_SIZE': round(min(max(random.normalvariate(3.5, 1), 2), 5)), # vary between 0 - 10%,
    }
    (grammar_depth_min, grammar_depth_max), grammar_n_non_terminals, (grammar_n_prods_occurrences, grammar_n_recursive_prods) = grammar.get_grammar_specifics()
    ffs, target_ind = ffs_and_target_ind

    for representation in [ "ge", "dsge", "treebased" ]:
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
        for ff_level, ff in ffs:    
            for max_depth in [6, 8, 10, 12, 14]:
                repr.method = Random_Production()

                def evolve(
                    seed,
                    mode,
                ):
                    print(grammar)

                    so_problem=SingleObjectiveProblem(
                                minimize=params['MINIMIZE'],
                                fitness_function=ff,
                                target_fitness=params['TARGET_FITNESS'],
                            )

                    alg = GP(
                        grammar,
                        representation=repr,
                        problem=so_problem,
                        probability_crossover=params['PROBABILITY_CO'],
                        probability_mutation=params['PROBABILITY_MUT'],
                        cross_over_return_one_individual=True,
                        number_of_generations=params['NUMBER_OF_ITERATIONS'],
                        max_init_depth=min(max_depth, params['MAX_INIT_DEPTH']),
                        max_depth=max_depth,
                        population_size=params['POPULATION_SIZE'],
                        selection_method=("tournament",params['TOURNAMENT_SIZE']),
                        n_elites=params['ELITSM'],
                        n_novelties=params['NOVELTY'],
                        save_to_csv=CSVCallback(
                            filename=f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/{ff_level}_d{max_depth}_s{seed}.csv",
                            save_genotype_as_string=True,
                            save_productions=True,
                            ),
                        seed=seed,
                        timer_stop_criteria=mode,
                    )
                    (b, bf, bp) = alg.evolve(verbose=1)
                    return b, bf, bp, b.count_prods(repr.genotype_to_phenotype, grammar)

                individual, fitness, phenotype, prods = evolve(seed, False)
                fitness = ff(phenotype)
                print(phenotype)
                print(f"With fitness: {fitness}")
                csv_row = [ fitness, seed, benchmark_name, individual.genotype, phenotype, prods, ff_level, target_ind, max_depth, params["MAX_INIT_DEPTH"], params["POPULATION_SIZE"], params["ELITSM"], params["TARGET_FITNESS"], params["PROBABILITY_CO"], params["PROBABILITY_MUT"], params["NOVELTY"], params["TOURNAMENT_SIZE"], grammar_depth_min, grammar_depth_max, grammar_n_non_terminals, grammar_n_prods_occurrences, grammar_n_recursive_prods ]
                with open(f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/main.csv", "a", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(csv_row)
    
