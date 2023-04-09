from __future__ import annotations
from argparse import ArgumentParser

from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.coding.classes import Condition
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Not
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.metahandlers.ints import IntRange

from examples.utils.wrapper import run_experiments


# This example is a spatial problem. Originally, we planned to implement the problem given in this paper: https://www.sciencedirect.com/science/article/pii/S030438000100309X?casa_token=NV39DBUNnswAAAAA:oRJp2wH7cwQp40tb4Mke90wZKIHjTtKBMF7zkSfyQ8N4_ZT_dDIjzP_ClCJoGid1fvSdZTvZIUU. We couldn't find the data nor the code, and authors weren't responsive. We also tried contacting the original publisher of the data. We implemented game of life as a subsitute.

DATASET_NAME = "GameOfLife"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"
OUTPUT_FOLDER = "GoL/grammar_standard"


train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",")
Xtrain = train[:, :-1]
Xtrain = Xtrain.reshape(train.shape[0], 3, 3)
ytrain = train[:, -1]

test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",")
Xtest = test[:, :-1]
Xtest = Xtest.reshape(test.shape[0], 3, 3)
ytest = test[:, -1]


@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, 2)]
    column: Annotated[int, IntRange(0, 2)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"


def evaluate(e: Expr) -> Callable[[Any], float]:
    if isinstance(e, And):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) and f2(line)
    elif isinstance(e, Or):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) or f2(line)
    elif isinstance(e, Not):
        f1 = evaluate(e.cond)
        return lambda line: not f1(line)
    elif isinstance(e, MatrixElement):
        r = e.row
        c = e.column
        return lambda line: line[r, c]
    else:
        raise NotImplementedError(str(e))


def fitness_function(i: Expr):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


def fitness_function_test(i: Expr):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    return f1_score(ytest, ypred)


grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)

params = {
    "MINIMIZE": False,
    "NUMBER_OF_ITERATIONS": 50,
    "MIN_INIT_DEPTH": 2,
    "MIN_DEPTH": None,
    "MAX_INIT_DEPTH": 6,
    "MAX_DEPTH": 10,
    "POPULATION_SIZE": 50,
    "ELITSM": 5,
    "TARGET_FITNESS": None,
}

if __name__ == "__main__":
    representations = ["ge", "dsge", "treebased"]

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()

    run_experiments(
        grammar,
        ff=fitness_function,
        ff_test=fitness_function_test,
        folder_name="game_of_life",
        seed=args.seed,
        params=params,
        representation=representations[args.representation],
    )
