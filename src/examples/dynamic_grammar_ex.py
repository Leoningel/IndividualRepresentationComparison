
from argparse import ArgumentParser

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource

from geneticengine.grammars.dynamic_grammar import create_grammar_nodes
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.dynamic_grammar import edit_distance
from geneticengine.algorithms.gp.individual import Individual

from examples.utils.wrapper import run_experiments

def grammar_and_ff_def(g_seed=321, n_class_abc=6, n_class_0_ch=4, n_class_2_ch=12, max_var_per_class=4, level_of_hardness='hard'):
    (list, starting_node) = create_grammar_nodes(
        seed=g_seed,
        n_class_abc = n_class_abc,
        n_class_0_children = n_class_0_ch,
        n_class_2_children = n_class_2_ch,
        max_var_per_class = max_var_per_class,
    )

    grammar = extract_grammar(list, starting_node)


    def create_target_individual(grammar_seed, grammar):
        r = RandomSource(grammar_seed)
        representation = treebased_representation
        target_individual = representation.create_individual(r=r, g=grammar, depth=10)
        individual_phenotype = representation.genotype_to_phenotype(
            grammar, target_individual)
        return individual_phenotype


    def fitness_function():
        random = RandomSource(g_seed)
        target_individual = create_target_individual(random.randint(0,999999), grammar)

        if level_of_hardness == 'hard':
            def ff(n):
                return edit_distance(str(n), str(target_individual))
        else:
            ind = Individual(target_individual)
            ti_prods = ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
            print(ti_prods)
            if level_of_hardness == 'medium':
                def ff(n):
                    n_ind = Individual(n)
                    prods = n_ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
                    prod_differences = 0
                    for prod in prods.keys():
                        prod_differences += abs(ti_prods[prod] - prods[prod])
                    if prod_differences == 0:
                        ti = target_individual
                        import IPython as ip
                        ip.embed()
                    return prod_differences
            elif level_of_hardness == 'easy':
                random_key = random.choice([key for key in ti_prods.keys()])
                def ff(n):
                    n_ind = Individual(n)
                    prods = n_ind.count_prods(treebased_representation.genotype_to_phenotype, grammar)
                    return abs(ti_prods[random_key] - prods[random_key])
        return ff
    return grammar, fitness_function
            

params = {
    'MINIMIZE': True,
    'NUMBER_OF_ITERATIONS': 25,
    'MIN_INIT_DEPTH': None,
    'MIN_DEPTH': None,
    'MAX_INIT_DEPTH': 4,
    'MAX_DEPTH': 8,
    'POPULATION_SIZE': 20,
    'ELITSM': 5,
    'TARGET_FITNESS': 0,
}

if __name__ == "__main__":
    representations = ['ge', 'dsge', 'treebased']

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation",
                        dest="representation", type=int, default=0)
    args = parser.parse_args()

