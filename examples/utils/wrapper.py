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
    ProgressCallback,
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
    target_individual: Any,
):
    print("Single run", benchmark_name, seed, representation, ff_level, max_depth)

    (
        (grammar_depth_min, grammar_depth_max),
        grammar_n_non_terminals,
        (grammar_n_prods_occurrences, grammar_n_recursive_prods, grammar_alternatives, grammar_total_productions, grammar_average_productions_per_terminal),
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
            # -- Grammar ------------------
            "Grammar Depth Min": lambda gen, pop, time, gp, ind: grammar_depth_min,  # Smallest possible individual's depth
            "Grammar Depth Max": lambda gen, pop, time, gp, ind: grammar_depth_max,  # Biggest possible individual's depth (max 10000)
            "Grammar Non Terminals": lambda gen, pop, time, gp, ind: grammar_n_non_terminals,  # Number of different non terminals (unique elements without children)
            "Grammar Productions Ocurrences Count": lambda gen, pop, time, gp, ind: grammar_n_prods_occurrences,  # Dictionary with: { symbol: number of times it occurs on the RHS }
            "Grammar Recursive Productions Count": lambda gen, pop, time, gp, ind: grammar_n_recursive_prods,  # Number of recursive productions
            "Grammar Productions Per Non Terminal": lambda gen, pop, time, gp, ind: grammar_alternatives,  # The alternatives/productions possible for each non-terminal.
            "Grammar Total Number of Productions": lambda gen, pop, time, gp, ind: grammar_total_productions,  # The total number of productions in the grammar.
            "Grammar Average Number of Productions": lambda gen, pop, time, gp, ind: grammar_average_productions_per_terminal,  # The average number of productions for each non-terminal.
            # -- Grammar Creation Variables ------------------
            "Requested Non Terminals Count": lambda gen, pop, time, gp, ind: non_terminals_count,  #
            "Requested Recursive Non Terminals Count": lambda gen, pop, time, gp, ind: recursive_non_terminals_count,  #
            "Requested Average Productions per Terminal": lambda gen, pop, time, gp, ind: average_productions_per_terminal,  #
            "Requested Non Terminals per Production": lambda gen, pop, time, gp, ind: non_terminals_per_production,  #
            # Target Individual Information
            "Target Individual": lambda gen, pop, time, gp, ind: str(target_individual),  #
            "Target Individual Nodes": lambda gen, pop, time, gp, ind: target_individual.gengy_nodes,  #
            "Target Individual Depth": lambda gen, pop, time, gp, ind: target_individual.gengy_distance_to_term,  #
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
    fitness_function: tuple[str, Any, Any],
    non_terminals_count: int,
    recursive_non_terminals_count: int,
    average_productions_per_terminal: int,
    non_terminals_per_production: int,
):
    params = make_synthetic_params(seed)
    representation_name, repr = make_representations()[representation_index]
    ff_level, ff, target_individual = fitness_function
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
            target_individual=target_individual,
        )
    except Exception:
        sys.stderr.write(traceback.format_exc())


def run_experiments(
    grammar,
    ff,
    ff_test,
    benchmark_name,
    seed,
    params,
    repr_code: int,
    timeout=5 * 60,
):
    """Runs an experiment run"""

    representation = [
        GrammaticalEvolutionRepresentation,
        DynamicStructuredGrammaticalEvolutionRepresentation,
        TreeBasedRepresentation,
    ][repr_code]

    representation_name = str(representation.__name__)
    max_depth = params.get("MAX_DEPTH", 6)
    population_size = params.get("POPULATION_SIZE", 20)
    elitism = params.get("ELITISM", 1)
    novelty = params.get("NOVELTY", 0)
    tournament_size = params.get("TOURNAMENT_SIZE", 3)
    probability_co = params.get("PROBABILITY_CO", 0.1)
    probability_mut = params.get("PROBABILITY_MUT", 0.9)
    minimize = params.get("MINIMIZE", False)
    target_fitness = params.get("TARGET_FITNESS", None)

    remaining = population_size - elitism - novelty

    so_problem = SingleObjectiveProblem(minimize=minimize, fitness_function=ff)
    os.makedirs(f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation_name}/", exist_ok=True)
    mcb = MemoryCallback()

    (
        (grammar_depth_min, grammar_depth_max),
        grammar_n_non_terminals,
        (grammar_n_prods_occurrences, grammar_n_recursive_prods, grammar_alternatives, grammar_total_productions, grammar_average_productions_per_terminal),
    ) = grammar.get_grammar_properties_summary()

    extra_columns = {
        "Phenotype": lambda gen, pop, time, gp, ind: str(ind.get_phenotype()),
        "GP Seed": lambda gen, pop, time, gp, ind: seed,
        "Benchmark Name": lambda gen, pop, time, gp, ind: benchmark_name,
        "Grammar": lambda gen, pop, time, gp, ind: grammar,
        "Representation": lambda gen, pop, time, gp, ind: representation_name,
        "Max Depth": lambda gen, pop, time, gp, ind: max_depth,
        "Mem Peak": lambda gen, pop, time, gp, ind: mcb.mem_peak,
        "Population Size": lambda gen, pop, time, gp, ind: population_size,
        "Elitism": lambda gen, pop, time, gp, ind: elitism,
        "Novelty": lambda gen, pop, time, gp, ind: novelty,
        "Probability Crossover": lambda gen, pop, time, gp, ind: probability_co,
        "Probability Mutation": lambda gen, pop, time, gp, ind: probability_mut,
        "Tournament Size": lambda gen, pop, time, gp, ind: tournament_size,
        # -- Grammar ------------------
        "Grammar Depth Min": lambda gen, pop, time, gp, ind: grammar_depth_min,  # Smallest possible individual's depth
        "Grammar Depth Max": lambda gen, pop, time, gp, ind: grammar_depth_max,  # Biggest possible individual's depth (max 10000)
        "Grammar Non Terminals": lambda gen, pop, time, gp, ind: grammar_n_non_terminals,  # Number of different non terminals (unique elements without children)
        "Grammar Productions Ocurrences Count": lambda gen, pop, time, gp, ind: grammar_n_prods_occurrences,  # Dictionary with: { symbol: number of times it occurs on the RHS }
        "Grammar Recursive Productions Count": lambda gen, pop, time, gp, ind: grammar_n_recursive_prods,  # Number of recursive productions
        "Grammar Productions Per Non Terminal": lambda gen, pop, time, gp, ind: grammar_alternatives,  # The alternatives/productions possible for each non-terminal.
        "Grammar Total Number of Productions": lambda gen, pop, time, gp, ind: grammar_total_productions,  # The total number of productions in the grammar.
        "Grammar Average Number of Productions": lambda gen, pop, time, gp, ind: grammar_average_productions_per_terminal,  # The average number of productions for each non-terminal.
    }

    if ff_test is not None:
        extra_columns["Test Fitness"] = lambda gen, pop, time, gp, ind: ff_test(ind.get_phenotype())

    csvcb = CSVCallback(
        filename=f"{gv.RESULTS_FOLDER}/{benchmark_name}/{representation_name}/s{seed}.csv",
        extra_columns=extra_columns,
    )

    step = ParallelStep(
        [
            ElitismStep(),
            NoveltyStep(),
            SequenceStep(
                TournamentSelection(tournament_size),
                GenericCrossoverStep(probability_co),
                GenericMutationStep(probability_mut),
            ),
        ],
        weights=[elitism, novelty, remaining],
    )

    stopping_criterium = TimeStoppingCriterium(timeout)
    if target_fitness is not None:
        stopping_criterium = AnyOfStoppingCriterium(
            stopping_criterium, SingleFitnessTargetStoppingCriterium(target_fitness)
        )

    alg = GP(
        representation=representation(grammar=grammar, max_depth=max_depth),
        problem=so_problem,
        random_source=RandomSource(seed),
        population_size=population_size,
        step=step,
        stopping_criterium=stopping_criterium,
        callbacks=[mcb, csvcb, ProgressCallback()],
    )

    alg.evolve()
