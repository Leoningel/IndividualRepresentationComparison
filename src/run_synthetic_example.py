from argparse import ArgumentParser
import random
import numpy as np
from examples.utils.wrapper import run_synthetic_experiments

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=123)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    from examples.synthetic_grammar_ex import grammar_and_ff_def

    for seedx in range(seed, seed + 30):
        try:
            grammar, ffs = grammar_and_ff_def(seed=seedx)
            run_synthetic_experiments(
                grammar, ffs_and_target_ind=ffs(), benchmark_name="synthetic", seed=seed
            )
            break
        except AssertionError:
            pass
