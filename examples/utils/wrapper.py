from __future__ import annotations
import os

from random import Random
import sys
import traceback

from typing import Any

from geneticengine.algorithms.gp.operators.stop import (
    AnyOfStoppingCriterium,
    SingleFitnessTargetStoppingCriterium,
)
from geneticengine.algorithms.hill_climbing import GenericMutationStep
from geneticengine.algorithms.random_mutations import (
    ElitismStep,
    ParallelStep,
    SequenceStep,
)
from geneticengine.algorithms.random_search import NoveltyStep
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.off_the_shelf.classifiers import TreeBasedRepresentation
from geneticengine.prelude import RandomSource

import examples.utils.global_vars as gv

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP, GenericCrossoverStep, TournamentSelection
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.algorithms.callbacks.callback import (
    Callback,
    TimeStoppingCriterium,
)

import platform


if platform.python_implementation() == "PyPy":

    class MemoryCallback(Callback):
        def __init__(self):
            self.mem_peak = 0

        def process_iteration(self, generation: int, population, time: float, gp):
            pass

        def end_evolution(self):
            self.mem_peak = 0

else:
    import tracemalloc

    class MemoryCallback(Callback):
        def __init__(self):
            tracemalloc.start()
            self.mem_peak = 0

        def process_iteration(self, generation: int, population, time: float, gp):
            pass

        def end_evolution(self):
            self.mem_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()


def single_run(
    base_seed,
    timeout,
    seed,
    params,
    grammar: Grammar,
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

    (
        (grammar_depth_min, grammar_depth_max),
        grammar_n_non_terminals,
        (grammar_n_prods_occurrences, grammar_n_recursive_prods),
    ) = grammar.get_grammar_properties_summary()

    so_problem = SingleObjectiveProblem(
        minimize=params["MINIMIZE"],
        fitness_function=ff,
    )
    os.makedirs(f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/", exist_ok=True)
    mcb = MemoryCallback()

    csvcb = CSVCallback(
        filename=f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation}/{ff_level}_d{max_depth}_s{seed}.csv",
        extra_columns={
            "Grammar Seed": lambda gen, pop, time, gp, ind: base_seed,
            "GP Seed": lambda gen, pop, time, gp, ind: seed,
            "Benchmark Name": lambda gen, pop, time, gp, ind: benchmark_name,
            "Grammar": lambda gen, pop, time, gp, ind: grammar,
            "Fitness Difficulty": lambda gen, pop, time, gp, ind: ff_level,
            "Representation": lambda gen, pop, time, gp, ind: str(repr.__name__),
            "Max Depth": lambda gen, pop, time, gp, ind: max_depth,
            "Mem Peak": lambda gen, pop, time, gp, ind: mcb.mem_peak,
            "Population Size": lambda gen, pop, time, gp, ind: params["POPULATION_SIZE"],
            "Elitism": lambda gen, pop, time, gp, ind: params["ELITISM"],
            "Novelty": lambda gen, pop, time, gp, ind: params["NOVELTY"],
            "Probability Crossover": lambda gen, pop, time, gp, ind: params["PROBABILITY_CO"],
            "Probability Mutation": lambda gen, pop, time, gp, ind: params["PROBABILITY_MUT"],
            "Tournament Size": lambda gen, pop, time, gp, ind: params["TOURNAMENT_SIZE"],
            "Grammar Depth Min": lambda gen, pop, time, gp, ind: grammar_depth_min,
            "Grammar Depth Max": lambda gen, pop, time, gp, ind: grammar_depth_max,
            "Grammar Non Terminals": lambda gen, pop, time, gp, ind: grammar_n_non_terminals,
            "Grammar Productions Ocurrences Count": lambda gen, pop, time, gp, ind: grammar_n_prods_occurrences,
            "Grammar Recursive Productions Count": lambda gen, pop, time, gp, ind: grammar_n_prods_occurrences,
            "Requested Non Terminals Count": lambda gen, pop, time, gp, ind: non_terminals_count,
            "Requested Recursive Non Terminals Count": lambda gen, pop, time, gp, ind: recursive_non_terminals_count,
            "Requested Average Productions per Terminal": lambda gen, pop, time, gp, ind: average_productions_per_terminal,
            "Requested Non Terminals per Production": lambda gen, pop, time, gp, ind: non_terminals_per_production,
        },
    )

    remaining = params["POPULATION_SIZE"] - params["ELITISM"] - params["NOVELTY"]

    step = ParallelStep(
        [
            ElitismStep(),
            NoveltyStep(),
            SequenceStep(
                TournamentSelection(params["TOURNAMENT_SIZE"]),
                GenericCrossoverStep(params["PROBABILITY_CO"]),
                GenericMutationStep(params["PROBABILITY_MUT"]),
            ),
        ],
        weights=[params["ELITISM"], params["NOVELTY"], remaining],
    )

    alg = GP(
        representation=repr(grammar=grammar, max_depth=max_depth),
        problem=so_problem,
        random_source=RandomSource(seed),
        population_size=params["POPULATION_SIZE"],
        step=step,
        stopping_criterium=AnyOfStoppingCriterium(
            SingleFitnessTargetStoppingCriterium(params["TARGET_FITNESS"]),  # TODO
            TimeStoppingCriterium(timeout),
        ),
        callbacks=[mcb, csvcb],
    )
    ind = alg.evolve()

    print(
        "seed",
        seed,
        "difficulty",
        ff_level,
        "version",
        representation,
        "grammar",
        grammar,
        "fitness",
        ind.get_fitness(so_problem),
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
        "ELITISM": round(min(max(random.normalvariate(5, 2), 0), pop_size / 10)),  # vary between 0 - 10%
        "TARGET_FITNESS": 0,
        "PROBABILITY_CO": min(1, max(0, random.normalvariate(0.6, 0.3))),
        "PROBABILITY_MUT": min(1, max(0, random.normalvariate(0.6, 0.3))),
        "NOVELTY": round(min(max(random.normalvariate(5, 2), 0), pop_size / 10)),  # vary between 0 - 10%
        "TOURNAMENT_SIZE": round(min(max(random.normalvariate(3.5, 1), 2), 5)),  # vary between 0 - 10%,
    }


def make_representations():
    ge_repr = GrammaticalEvolutionRepresentation

    dsge_repr = DynamicStructuredGrammaticalEvolutionRepresentation

    tree_repr = TreeBasedRepresentation

    return [("ge", ge_repr), ("dsge", dsge_repr), ("treebased", tree_repr)]


def run_synthetic_experiments(
    benchmark_name: str,
    base_seed: int,
    timeout: int,
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
            base_seed,
            timeout,
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
