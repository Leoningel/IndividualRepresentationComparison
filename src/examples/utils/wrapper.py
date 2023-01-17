from __future__ import annotations
import csv
import os

from random import Random
import sys
import time
import traceback

from typing import Any

import examples.utils.global_vars as gv

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (  # noqa: E501
    dsge_representation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.tree.initialization_methods import (
    Random_Production,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.algorithms.callbacks.callback import Callback

import platform

if platform.python_implementation() == "PyPy":

    class MemoryCallback(Callback):
        def process_iteration(self, generation: int, population, time: float, gp):
            pass

        def end_evolution(self):
            self.mem_peak = 0

else:
    import tracemalloc

    class MemoryCallback(Callback):
        def __init__(self):
            tracemalloc.start()

        def process_iteration(self, generation: int, population, time: float, gp):
            pass

        def end_evolution(self):
            self.mem_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()


class GenerationCallback(Callback):
    def __init__(self):
        self.generations = 0
        self.first_generation_fitness = None

    def process_iteration(self, generation: int, population, time: float, gp: GP):
        self.generations = generation
        if self.first_generation_fitness is None:
            self.first_generation_fitness = gp.get_best_individual(
                gp.problem, population
            )

    def end_evolution(self):
        pass


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
        repr.mutation_method = "per_codon_mutate"
        repr.codon_prob = 0.05
    elif representation == "dsge":
        repr = dsge_representation
        repr.mutation_method = "per_codon_mutate"
        repr.codon_prob = 0.05
    else:
        repr = treebased_representation
        novelty = int(params["POPULATION_SIZE"] * 0.05)
    repr.method = Random_Production()

    def evolve(
        seed,
        mode,
    ):
        print(grammar)

        so_problem = SingleObjectiveProblem(
            minimize=params["MINIMIZE"],
            fitness_function=ff,
            target_fitness=params["TARGET_FITNESS"],
        )

        alg = GP(
            grammar,
            representation=repr,
            problem=so_problem,
            probability_crossover=gv.PROB_CROSSOVER,
            probability_mutation=gv.PROB_MUTATION,
            cross_over_return_one_individual=True,
            number_of_generations=params["NUMBER_OF_ITERATIONS"],
            min_init_depth=params["MIN_INIT_DEPTH"],
            min_depth=params["MIN_DEPTH"],
            max_init_depth=params["MAX_INIT_DEPTH"],
            max_depth=params["MAX_DEPTH"],
            population_size=params["POPULATION_SIZE"],
            selection_method=("tournament", gv.TOURNAMENT),
            n_elites=params["ELITSM"],
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
    csv_row = [
        fitness,
        test_fitness,
        seed,
        benchmark_name,
        individual.genotype,
        phenotype,
        prods,
    ]
    with open(
        f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/main.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)


def single_run(
    seed,
    params,
    grammar,
    benchmark_name,
    representation,
    repr,
    ff_level,
    ff,
    max_depth,
    non_terminals_count: int,
    recursive_non_terminals_count: int,
    average_productions_per_terminal: int,
    non_terminals_per_production: int,
):
    print("Single run", benchmark_name, seed, representation, ff_level, max_depth)
    repr.method = Random_Production()

    so_problem = SingleObjectiveProblem(
        minimize=params["MINIMIZE"],
        fitness_function=ff,
        target_fitness=params["TARGET_FITNESS"],
    )
    os.makedirs(
        f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/", exist_ok=True
    )
    mcb = MemoryCallback()
    gcb = GenerationCallback()
    start_time = time.time()
    alg = GP(
        grammar,
        representation=repr,
        problem=so_problem,
        probability_crossover=params["PROBABILITY_CO"],
        probability_mutation=params["PROBABILITY_MUT"],
        cross_over_return_one_individual=True,
        number_of_generations=params["NUMBER_OF_ITERATIONS"],
        # max_init_depth=min(max_depth, params["MAX_INIT_DEPTH"]),
        max_depth=max_depth,
        population_size=params["POPULATION_SIZE"],
        selection_method=("tournament", params["TOURNAMENT_SIZE"]),
        n_elites=params["ELITSM"],
        n_novelties=params["NOVELTY"],
        save_to_csv=CSVCallback(
            filename=f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/{ff_level}_d{max_depth}_s{seed}.csv",
            save_genotype_as_string=True,
            save_productions=False,
        ),
        callbacks=[mcb, gcb],
        seed=seed,
        timer_stop_criteria=False,
        target_fitness=0,
    )
    (individual, fitness, phenotype) = alg.evolve(verbose=1)
    end_time = time.time() - start_time
    fitness = ff(phenotype)
    (
        (grammar_depth_min, grammar_depth_max),
        grammar_n_non_terminals,
        (grammar_n_prods_occurrences, grammar_n_recursive_prods),
    ) = grammar.get_grammar_specifics()

    csv_row = [
        seed,
        benchmark_name,
        grammar,
        ff_level,
        representation,
        max_depth,
        fitness,
        end_time,
        mcb.mem_peak,
        gcb.generations,
        gcb.first_generation_fitness,
        # individual.genotype,
        # phenotype,
        # params["MAX_INIT_DEPTH"],
        params["POPULATION_SIZE"],
        params["ELITSM"],
        params["PROBABILITY_CO"],
        params["PROBABILITY_MUT"],
        params["NOVELTY"],
        params["TOURNAMENT_SIZE"],
        grammar_depth_min,
        grammar_depth_max,
        grammar_n_non_terminals,
        grammar_n_prods_occurrences,
        grammar_n_recursive_prods,
        non_terminals_count,
        recursive_non_terminals_count,
        average_productions_per_terminal,
        non_terminals_per_production,
    ]
    with open(
        f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/main.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)

    print(
        "difficulty",
        ff_level,
        "version",
        representation,
        "generations",
        gcb.generations,
        "fitness",
        fitness,
        "max_time",
        end_time,
        "max_memory",
        mcb.mem_peak,
        "non_terminals",
        non_terminals_count,
        "recursive non terminsl",
        recursive_non_terminals_count,
        "average_prod_per_terminal",
        average_productions_per_terminal,
        "non_terminals_per_production",
        non_terminals_per_production,
    )


def make_synthetic_params(seed: int):
    random = Random(seed)
    pop_size = round(random.normalvariate(50, 5))

    return {
        "MINIMIZE": True,
        "NUMBER_OF_ITERATIONS": 100,
        "MAX_INIT_DEPTH": round(random.normalvariate(5, 1.5)),  # vary
        "POPULATION_SIZE": pop_size,  # vary
        "ELITSM": round(
            min(max(random.normalvariate(5, 2), 0), pop_size / 10)
        ),  # vary between 0 - 10%
        "TARGET_FITNESS": 0,
        "PROBABILITY_CO": min(1, max(0, random.normalvariate(0.6, 0.3))),
        "PROBABILITY_MUT": min(1, max(0, random.normalvariate(0.6, 0.3))),
        "NOVELTY": round(
            min(max(random.normalvariate(5, 2), 0), pop_size / 10)
        ),  # vary between 0 - 10%
        "TOURNAMENT_SIZE": round(
            min(max(random.normalvariate(3.5, 1), 2), 5)
        ),  # vary between 0 - 10%,
    }


def make_representations():
    ge_repr = ge_representation
    ge_repr.mutation_method = "per_codon_mutate"
    ge_repr.codon_prob = 0.05

    dsge_repr = dsge_representation
    dsge_repr.mutation_method = "per_codon_mutate"
    dsge_repr.codon_prob = 0.05

    tree_repr = treebased_representation

    return [("ge", ge_repr), ("dsge", dsge_repr), ("treebased", tree_repr)]


def run_synthetic_experiments(
    benchmark_name: str,
    seed: int,
    grammar: Grammar,
    representation_index: int,
    target_individual,
    target_depth: int,
    fitness_function: tuple[str, Any],
    non_terminals_count: int,
    recursive_non_terminals_count: int,
    average_productions_per_terminal: int,
    non_terminals_per_production: int,
):
    params = make_synthetic_params(seed)
    representation_name, repr = make_representations()[representation_index]
    ff_level, ff = fitness_function
    try:
        single_run(
            seed,
            params,
            grammar,
            benchmark_name,
            representation_name,
            repr,
            ff_level,
            ff,
            target_depth,
            non_terminals_count=non_terminals_count,
            recursive_non_terminals_count=recursive_non_terminals_count,
            average_productions_per_terminal=average_productions_per_terminal,
            non_terminals_per_production=non_terminals_per_production,
        )
    except Exception:
        sys.stderr.write(traceback.format_exc())
