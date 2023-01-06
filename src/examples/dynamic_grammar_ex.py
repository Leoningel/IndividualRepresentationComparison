
from argparse import ArgumentParser
from polyleven import levenshtein


from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.synthetic_grammar import create_arbitrary_grammar

from examples.utils.wrapper import run_experiments

def grammar_and_ff_def(seed=123, non_terminals_count=6, recursive_non_terminals_count=2):
    (list, starting_node) = create_arbitrary_grammar(
        seed=seed,
        non_terminals_count=non_terminals_count,
        recursive_non_terminals_count=recursive_non_terminals_count,
    )

    grammar = extract_grammar(list, starting_node)


    def create_target_individual(grammar_seed, grammar):
        r = RandomSource(grammar_seed)
        representation = treebased_representation
        target_individual = representation.create_individual(r=r, g=grammar, depth=10)
        individual_phenotype = representation.genotype_to_phenotype(
            grammar, target_individual)
        return individual_phenotype


    def fitness_functions():
        random = RandomSource(seed)
        target_individual = create_target_individual(random.randint(0,999999), grammar)

        def ff_hard(n):
            return levenshtein(str(n), str(target_individual))
        ind = Individual(target_individual)
        ti_prods = ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
        def ff_medium(n):
            n_ind = Individual(n)
            prods = n_ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
            prod_differences = 0
            for prod in prods.keys():
                prod_differences += abs(ti_prods[prod] - prods[prod])
            return prod_differences
        random_key = random.choice([key for key in ti_prods.keys()])
        def ff_easy(n):
            n_ind = Individual(n)
            prods = n_ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
            return abs(ti_prods[random_key] - prods[random_key])
        return [ ("easy", ff_easy), ("medium", ff_medium), ("hard", ff_hard) ], target_individual
    return grammar, fitness_functions
            


