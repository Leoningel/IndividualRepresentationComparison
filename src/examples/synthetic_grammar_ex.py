import random


import platform

if platform.python_implementation() == "PyPy":
    from pylev import levenshtein
else:
    from polyleven import levenshtein


from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.tree.initialization_methods import (
    Random_Production,
)
from geneticengine.grammars.synthetic_grammar import create_arbitrary_grammar


def generate_synthetic_grammar(
    seed: int,
    non_terminals_count: int = 6,
    recursive_non_terminals_count: int = 2,
    average_productions_per_terminal: int = 10,
    non_terminals_per_production: int = 10,
):
    """Generates an arbitratry gramar."""
    (list, starting_node) = create_arbitrary_grammar(
        seed=seed,
        non_terminals_count=non_terminals_count,
        recursive_non_terminals_count=recursive_non_terminals_count,
        productions_per_non_terminal=lambda rd: round(
            rd.uniform(1, average_productions_per_terminal),
        ),
        non_terminals_per_production=lambda rd: round(
            rd.uniform(0, non_terminals_per_production),
        ),
    )
    grammar = extract_grammar(list, starting_node)
    return grammar


def validate_grammar(grammar) -> bool:
    """Returns whether a grammar is interesting for these experiments"""
    return all(
        [
            grammar.get_min_tree_depth() > 1,
            grammar.get_min_tree_depth() < 20,
        ]
    )


def generate_target_individual(seed: int, grammar, target_depth: int = 10):
    """Generates an arbitratry individual that uses a grammar."""
    r = RandomSource(seed)
    representation = treebased_representation
    representation.method = Random_Production(min_depth=target_depth)
    target_individual = representation.create_individual(
        r=r, g=grammar, depth=target_depth
    )
    individual_phenotype = representation.genotype_to_phenotype(
        grammar, target_individual
    )
    return individual_phenotype


def generate_fitness_functions(grammar, target_individual):
    """Generates three fitness_functions for a particular individual"""

    target_str = str(target_individual)

    def ff_hard_original(n):
        return levenshtein(str(n), target_str)

    def ff_hard(n):
        x = str(n)
        y = str(target_individual)
        return sum(0 if a != b else 1 for a, b in zip(x, y)) + abs(len(x) - len(y))

    ind = Individual(target_individual)
    ti_prods = ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)

    def ff_medium(n):
        n_ind = Individual(n)
        prods = n_ind.count_prods(
            treebased_representation.genotype_to_phenotype, grammar
        )
        prod_differences = 0
        for prod in prods.keys():
            prod_differences += abs(ti_prods[prod] - prods[prod])
        return prod_differences

    random_key = random.choice([key for key in ti_prods.keys()])

    def ff_easy(n):
        n_ind = Individual(n)
        prods = n_ind.count_prods(
            treebased_representation.genotype_to_phenotype, grammar
        )
        return abs(ti_prods[random_key] - prods[random_key])

    return [("easy", ff_easy), ("medium", ff_medium), ("hard", ff_hard)]


def generate_problem(seed: int, target_depth: int):
    """Generates a valid grammar, target_individual and fitness_functions."""
    seedx = seed
    for _ in range(30):
        non_terminals_count = random.randint(3, 20)
        recursive_non_terminals_count = random.randint(0, non_terminals_count - 1)
        average_productions_per_terminal = random.randint(0, 10)
        non_terminals_per_production = random.randint(0, 10)
        grammar = generate_synthetic_grammar(
            seedx,
            non_terminals_count=non_terminals_count,
            recursive_non_terminals_count=recursive_non_terminals_count,
            average_productions_per_terminal=average_productions_per_terminal,
            non_terminals_per_production=non_terminals_per_production,
        )
        if (
            not validate_grammar(grammar)
            or grammar.get_min_tree_depth() >= target_depth
        ):
            seedx += 10000
            print(f"Fail {seedx}")
            continue
        else:
            target_ind = generate_target_individual(seedx, grammar, target_depth)
            fitness_functions = generate_fitness_functions(grammar, target_ind)
            return (
                seedx,
                grammar,
                target_ind,
                target_depth,
                fitness_functions,
                non_terminals_count,
                recursive_non_terminals_count,
                average_productions_per_terminal,
                non_terminals_per_production,
            )
