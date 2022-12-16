from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntList, IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.random.sources import RandomSource

from geneticengine.grammars.dynamic_grammar import create_grammar_nodes
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.dynamic_grammar import edit_distance


#example not currently working because genetic engine 0.0.46 does not contain the implementation of dynamic grammars

from examples.utils.wrapper import run_experiments
import examples.utils.global_vars as gv

grammar_seed= 321;
(list, starting_node) = create_grammar_nodes(
    grammar_seed=grammar_seed,
    n_class_abc = 6,
    n_class_0_children = 4,
    n_class_2_children = 12,
    max_var_per_class = 4 ,
)

grammar = extract_grammar(list, starting_node)


def create_target_individual(grammar_seed, g):

    r = RandomSource(grammar_seed)
    representation = TreeBasedRepresentation
    target_individual = representation.create_individual(r, g, depth=10)
    individual_phenotype = representation.genotype_to_phenotype(
        g, target_individual)
    return individual_phenotype


target_individual = create_target_individual(grammar_seed, grammar)

def fitness_function(n):
    return edit_distance(str(n), str(target_individual))


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

    run_experiments(grammar, ff=fitness_function, ff_test=None, folder_name="dynamic_grammar", seed=args.seed, params=params, representation=representations[args.representation])
