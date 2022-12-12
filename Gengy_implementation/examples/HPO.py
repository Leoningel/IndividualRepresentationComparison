from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Annotated
import pandas as pd

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.metahandlers.floats import FloatRange, FloatList
from geneticengine.metahandlers.ints import IntList, IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange

from utils.wrapper import run_experiments
import utils.global_vars as gv


@abstract
class Start():
    pass
    
@abstract
class Model():
    pass

class LinearM(Model):
    pass

@dataclass
class NNet(Model):
    hidden_layers: Annotated[int, IntList([4, 8, 16])]

@abstract
class Kernel():
    pass

class Linear(Kernel):
    pass

@dataclass
class Polynomial(Kernel):
    degree: Annotated[int, IntRange(1, 5)]

@dataclass
class Radial(Kernel):
    gamma: Annotated[float, FloatList(0.1, 0.2, 0.5, 1)]

@dataclass
class SVM(Model):
    kernel: Kernel
    cost: Annotated[float, FloatList(0.1, 1, 10, 100, 1000)]

@abstract
class Features():
    pass

@dataclass
class FeatureList(Features):
    time: bool
    chick: bool
    diet: bool

@dataclass
class Solution(Start):
    model: Model
    features: Features

grammar = extract_grammar(
                considered_subtypes=[Start, Solution, Model, Features, FeatureList, LinearM, NNet, Kernel, Linear, Polynomial, Radial, SVM ],
                starting_symbol=Start,
                )

def evaluate(elem: Solution, data):
    return None


def fitness_function(data):
    def ff(ind: Start):
        model = evaluate(ind, data)
        if not model:
            return 999999999999999999
        else:
            model.fit(
                data[0], 
                data[1],
            )

        return 1
    return ff

data = pd.read_csv('data/ChickWeight.csv')
import IPython as ip
ip.embed()

if __name__ == "__main__":
    representations = [ 'ge', 'dsge', 'treebased' ]
    
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()

    run_experiments(grammar, ff=fitness_function(data), ff_test=fitness_function(data), folder_name="cnn", seed=args.seed, representation=representations[args.representation])




