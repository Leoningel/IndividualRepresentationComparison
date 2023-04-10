from argparse import ArgumentParser
from examples.utils.wrapper import run_experiments


if __name__ == "__main__":
    representations = ["ge", "dsge", "treebased"]

    parser = ArgumentParser()
    parser.add_argument("-e", "--example", dest="example", type=str)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()

    example = args.example
    if example == "cnn":
        from examples.CNN import (
            fitness_function,
            grammar,
            data_train,
            data_test,
            params,
        )

        ff_train = fitness_function(data_train)
        ff_test = fitness_function(data_test)
    elif example == "hpo":
        from examples.HPO import (
            fitness_function,
            grammar,
            data_train,
            data_test,
            params,
        )

        ff_train = fitness_function(data_train)
        ff_test = fitness_function(data_test)
    elif example == "santafe":
        from examples.santafe import fitness_function, grammar, params

        ff_train = fitness_function
        ff_test = None
    elif example == "game_of_life":
        from examples.game_of_life import (
            fitness_function,
            grammar,
            fitness_function_test,
            params,
        )

        ff_train = fitness_function
        ff_test = fitness_function_test
    elif example == "boston_housing":
        from examples.boston_housing import (
            fitness_function,
            grammar,
            data_train,
            data_test,
            params,
        )

        ff_train = fitness_function(data_train)
        ff_test = fitness_function(data_test)
    else:
        raise Exception(
            f"The example {example} is not included. Included examples: cnn, hpo, game_of_life, santafe, boston_housing"
        )

    run_experiments(
        grammar,
        ff=ff_train,
        ff_test=ff_test,
        benchmark_name=example,
        seed=args.seed,
        params=params,
        repr_code=args.representation,
    )
