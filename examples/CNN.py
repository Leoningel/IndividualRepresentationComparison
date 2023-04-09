from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Annotated
import functools

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntList, IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange

from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
import tensorflow as tf

from examples.utils.wrapper import run_experiments
import examples.utils.global_vars as gv

from examples.utils.cifar10http import load_data_alt


# This example follows the grammar from https://github.com/rhrlima/cbioge/blob/1d1ffc3b3d6c79d08524e4ab8ca5526815b0d38b/cbioge/assets/grammars/cnn_example2.json
# This is based on, but not exactly the same as, the grammar given in https://link.springer.com/article/10.1007/s10710-022-09432-0


@abstract
class Start:
    pass


@abstract
class Layer(ABC):
    pass


@dataclass(unsafe_hash=True)
class CNN(Start):
    layers: Annotated[list[Layer], ListSizeBetween(1, 3)]


@abstract
class LType(Layer):
    pass


@dataclass(unsafe_hash=True)
class LTypeLayer(Layer):
    l_type: LType
    layer: Layer


@abstract
class Parameter:
    def evaluate(self):
        raise ValueError("No evaluate method implemented")


@dataclass(unsafe_hash=True)
class PoolType(Parameter):
    pooltype: Annotated[str, VarRange(["maxpool", "avgpool"])]

    def evaluate(self):
        return self.pooltype


@dataclass(unsafe_hash=True)
class Filters(Parameter):
    filters: Annotated[int, IntList([16, 32, 64, 128, 256, 512])]

    def evaluate(self):
        return self.filters


@dataclass(unsafe_hash=True)
class Units(Parameter):
    units: Annotated[int, IntList([32, 64, 128, 256, 512, 1024])]

    def evaluate(self):
        return self.units


@dataclass(unsafe_hash=True)
class Ksize(Parameter):
    ksize: Annotated[int, IntRange(1, 5)]

    def evaluate(self):
        return self.ksize


@dataclass(unsafe_hash=True)
class Strides(Parameter):
    strides: Annotated[int, IntRange(1, 2)]

    def evaluate(self):
        return self.strides


@dataclass(unsafe_hash=True)
class Rate(Parameter):
    rate: Annotated[float, FloatRange(0, 0.5)]

    def evaluate(self):
        return self.rate


@dataclass(unsafe_hash=True)
class Padding(Parameter):
    padding: Annotated[str, VarRange(["valid", "same"])]

    def evaluate(self):
        return self.padding


@dataclass(unsafe_hash=True)
class Activation(Parameter):
    activation: Annotated[str, VarRange(["relu", "selu", "elu", "tanh", "sigmoid", "linear"])]

    def evaluate(self):
        return self.activation


@dataclass(unsafe_hash=True)
class ConvL(LType):
    filters: Filters
    ksize: Ksize
    strides: Strides
    padding: Padding
    activation: Activation


@dataclass(unsafe_hash=True)
class DenseL(LType):
    units: Units
    activation: Activation


@dataclass(unsafe_hash=True)
class DropoutL(LType):
    rate: Rate


@dataclass(unsafe_hash=True)
class PoolL(LType):
    pool_type: PoolType
    ksize: Ksize
    strides: Strides
    padding: Padding


@abstract
class EmptyL(LType):
    pass


grammar = extract_grammar(
    considered_subtypes=[
        Start,
        Layer,
        CNN,
        LType,
        LTypeLayer,
        Parameter,
        PoolType,
        Filters,
        Units,
        Ksize,
        Strides,
        Rate,
        Padding,
        Activation,
        ConvL,
        DenseL,
        DropoutL,
        PoolL,
        EmptyL,
    ],
    starting_symbol=Start,
)


def evaluate(elem: Start, dataset):
    def eval(elem):
        if isinstance(elem, Parameter):
            return elem.evaluate()
        if isinstance(elem, LTypeLayer):
            return (eval(elem.l_type), eval(elem.layer))
        if isinstance(elem, ConvL):
            conv_layer = Conv2D(
                filters=eval(elem.filters),
                kernel_size=eval(elem.ksize),
                strides=eval(elem.strides),
                padding=eval(elem.padding),
                activation=eval(elem.activation),
            )
            return conv_layer
        if isinstance(elem, DenseL):
            dense_layer = Dense(
                units=eval(elem.units),
                activation=eval(elem.activation),
            )
            return dense_layer
        if isinstance(elem, DropoutL):
            dense_layer = Dropout(
                rate=eval(elem.rate),
            )
            return dense_layer
        if isinstance(elem, PoolL):
            pool = MaxPooling2D if elem.pool_type == "maxpool" else AveragePooling2D
            pool_layer = pool(
                pool_size=eval(elem.ksize),
                strides=eval(elem.strides),
                padding=eval(elem.padding),
            )
            return pool_layer

    X = dataset[0]
    y = dataset[1]
    layers = list()
    layers.append(Input(shape=X[0].shape))

    def append(layer):
        if type(layer) == tuple:
            for l in layer:
                append(l)
        else:
            layers.append(layer)

    if isinstance(elem, CNN):
        for layer in elem.layers:
            append(eval(layer))
    else:
        raise Exception("The starting element should be CNN")

    layers.append(Flatten())
    layers.append(Dense(y.shape[1], activation="softmax"))
    try:
        # connecting the layers (functional API)
        in_layer = layers[0]
        out_layer = layers[0]
        for l in layers[1:]:
            out_layer = l(out_layer)

        model = Model(inputs=in_layer, outputs=out_layer)
    except ValueError:
        print("Invalid model")
        return None
        # raise Exception('Invalid model')
    return model


(X_train, y_train), (
    X_test,
    y_test,
) = load_data_alt()  # tf.keras.datasets.cifar10.load_data()
num_classes = len(set(y_train.transpose()[0]))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
data_train, data_test = (X_train[: gv.TRAIN_SIZE], y_train[: gv.TRAIN_SIZE]), (
    X_test[: gv.TEST_SIZE],
    y_test[: gv.TEST_SIZE],
)


def fitness_function(data):
    @functools.lru_cache
    def ff(ind: Start):
        model = evaluate(ind, data_train)
        if not model:
            return 999999999999999999
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="categorical_crossentropy",
            )
            es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=4, min_delta=0.1)
            model.fit(
                data_train[0],
                data_train[1],
                batch_size=gv.BATCH,
                epochs=gv.EPOCHS,
                verbose=0,
                callbacks=[es],
            )

        return model.evaluate(data[0], data[1])

    return ff


params = {
    "MINIMIZE": True,
    "NUMBER_OF_ITERATIONS": 25,
    "MIN_INIT_DEPTH": None,
    "MIN_DEPTH": None,
    "MAX_INIT_DEPTH": 4,
    "MAX_DEPTH": 8,
    "POPULATION_SIZE": 20,
    "ELITSM": 5,
    "TARGET_FITNESS": 0,
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
        benchmark_name="cnn",
        seed=args.seed,
        params=params,
        representation=representations[args.representation],
    )
