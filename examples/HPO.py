from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Annotated
import pandas as pd

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.metahandlers.floats import FloatList
from geneticengine.metahandlers.ints import IntList, IntRange

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import warnings


from examples.utils.wrapper import run_experiments


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# This example is based on the grammar given in https://www.jstatsoft.org/article/view/v071i01


@abstract
class Start:
    pass


@abstract
class Model:
    pass


@abstract
class LinearM(Model):
    pass


@dataclass
class NNet(Model):
    hidden_layers: Annotated[int, IntList([4, 8, 16])]


@abstract
class Kernel:
    pass


@abstract
class Linear(Kernel):
    pass


@dataclass
class Polynomial(Kernel):
    degree: Annotated[int, IntRange(1, 5)]


@dataclass
class Radial(Kernel):
    gamma: Annotated[float, FloatList([0.1, 0.2, 0.5, 1])]


@dataclass
class SVM(Model):
    kernel: Kernel
    cost: Annotated[float, FloatList([0.1, 1, 10, 100, 1000])]


@abstract
class Features:
    pass


@dataclass
class FeatureList(Features):
    time: Annotated[int, IntList([0, 1])]
    chick: Annotated[int, IntList([0, 1])]
    diet: Annotated[int, IntList([0, 1])]


@dataclass
class Solution(Start):
    model: Model
    features: FeatureList


grammar = extract_grammar(
    considered_subtypes=[
        Start,
        Solution,
        Model,
        Features,
        FeatureList,
        LinearM,
        NNet,
        Kernel,
        Linear,
        Polynomial,
        Radial,
        SVM,
    ],
    starting_symbol=Start,
)

data = pd.read_csv("examples/data/ChickWeight.csv")
X = data[["Time", "Chick", "Diet"]]
y = data["weight"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
data_train = (X_train, y_train)
data_test = (X_test, y_test)


def evaluate(elem: Solution, data_test):
    features = list()
    if bool(elem.features.time):
        features.append("Time")
    if bool(elem.features.chick):
        features.append("Chick")
    if bool(elem.features.diet):
        features.append("Diet")
    if not features:
        return None
    else:
        X_train = data_train[0][features].values
        y_train = data_train[1].values
        X_test = data_test[0][features].values
        y_test = data_test[1].values
    if isinstance(elem.model, LinearM):
        model = LinearRegression()
    if isinstance(elem.model, NNet):
        model = MLPRegressor(hidden_layer_sizes=elem.model.hidden_layers)
    if isinstance(elem.model, SVM):
        kernel = elem.model.kernel
        if isinstance(kernel, Linear):
            model = SVR(C=elem.model.cost, kernel="linear")
        if isinstance(kernel, Polynomial):
            model = SVR(C=elem.model.cost, kernel="poly", degree=kernel.degree)
        if isinstance(kernel, Radial):
            model = SVR(C=elem.model.cost, kernel="rbf", gamma=kernel.gamma)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def fitness_function(data):
    def ff(ind: Start):
        score = evaluate(ind, data)
        if not score:
            return -999999999999999999
        else:
            return score

    return ff


params = {
    "MINIMIZE": False,
    "NUMBER_OF_ITERATIONS": 25,
    "MIN_INIT_DEPTH": None,
    "MIN_DEPTH": None,
    "MAX_INIT_DEPTH": None,
    "MAX_DEPTH": 10,
    "POPULATION_SIZE": 20,
    "ELITISM": 1,
    "TARGET_FITNESS": 1,
}

if __name__ == "__main__":
    representations = ["ge", "dsge", "treebased"]

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()

    run_experiments(
        grammar,
        ff=fitness_function(data_train),
        ff_test=fitness_function(data_test),
        folder_name="hpo",
        seed=args.seed,
        params=params,
        representation=representations[args.representation],
    )
