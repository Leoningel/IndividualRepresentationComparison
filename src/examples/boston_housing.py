from __future__ import annotations
from argparse import ArgumentParser

import os
from dataclasses import dataclass
from math import isinf
from sklearn.model_selection import train_test_split
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    dsge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import Exp
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.basic_math import Sin
from geneticengine.grammars.basic_math import Tanh
from geneticengine.grammars.sgp import Minus
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse

from examples.utils.wrapper import run_experiments
import warnings

warnings.filterwarnings("ignore")


# This example is based on the grammar given in https://link.springer.com/chapter/10.1007/978-3-319-55696-3_20

DATA_FILE = f"examples/data/housing.csv"

bunch = pd.read_csv(DATA_FILE, delimiter=",")
y = bunch["MEDV"]
X = bunch.drop(["MEDV"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
data_train = (X_train, y_train)
data_test = (X_test, y_test)

feature_names = list(X_train.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


grammar = extract_grammar(
    [
        Plus,
        Minus,
        Mul,
        SafeDiv,
        Literal,
        Var,
        SafeSqrt,
        Exp,
        Sin,
        Tanh,
        SafeLog,
    ],
    Number,
)

# <e>  ::=  <e>+<e>|
#       <e>-<e>|
#       <e>*<e>|
#       pdiv(<e>,<e>)|
#       psqrt(<e>)|
#       np.sin(<e>)|
#       np.tanh(<e>)|
#       np.exp(<e>)|
#       plog(<e>)|
#       x[:, 0]|x[:, 1]|x[:, 2]|x[:, 3]|x[:, 4]|
#       <c>
# <c>  ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9


def fitness_function(data):
    def ff(n: Number):
        X = data[0].values
        y = data[1].values

        variables = {}
        for x in feature_names:
            i = feature_indices[x]
            variables[x] = X[:, i]

        y_pred = n.evaluate(**variables)
        # mse is used in PonyGE, as the error metric is not None!
        fitness = mse(y_pred, y)
        if isinf(fitness) or np.isnan(fitness):
            fitness = 100000000
        return fitness

    return ff


params = {
    "MINIMIZE": True,
    "NUMBER_OF_ITERATIONS": 50,
    "MIN_INIT_DEPTH": 2,
    "MIN_DEPTH": None,
    "MAX_INIT_DEPTH": 6,
    "MAX_DEPTH": 10,
    "POPULATION_SIZE": 200,
    "ELITISM": 5,
    "TARGET_FITNESS": 0,
}

if __name__ == "__main__":
    representations = ["ge", "dsge", "treebased"]

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument(
        "-r", "--representation", dest="representation", type=int, default=0
    )
    args = parser.parse_args()

    run_experiments(
        grammar,
        ff=fitness_function(data_train),
        ff_test=fitness_function(data_test),
        folder_name="boston_housing",
        seed=args.seed,
        params=params,
        representation=representations[args.representation],
    )
