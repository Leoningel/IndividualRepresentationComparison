
from argparse import ArgumentParser
from examples.utils.wrapper import run_synthetic_experiments

from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.grammatical_evolution.ge import ge_representation
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import dsge_representation

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=123)
    args = parser.parse_args()
    seed = args.seed

    from examples.dynamic_grammar_ex import grammar_and_ff_def
    grammar, ffs = grammar_and_ff_def(seed=seed)
    
    run_synthetic_experiments(grammar, ffs_and_target_ind=ffs(), benchmark_name=f"synthetic", seed=seed)
