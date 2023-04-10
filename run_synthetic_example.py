from argparse import ArgumentParser
import random
from examples.synthetic_grammar_ex import generate_problem
from examples.utils.wrapper import run_synthetic_experiments

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=123)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    parser.add_argument("-d", "--depth", dest="depth", type=int, default=0)
    parser.add_argument("-f", "--fitness", dest="fitness", type=int, default=0)
    parser.add_argument("-t", "--timeout", dest="timeout", type=int, default=5 * 60)

    args = parser.parse_args()
    base_seed = args.seed
    random.seed(base_seed)

    target_depth = [8, 10, 12, 14, 16][args.depth]

    (
        seed,
        grammar,
        target_individual,
        target_depth,
        fitness_functions,
        non_terminals_count,
        recursive_non_terminals_count,
        average_productions_per_terminal,
        non_terminals_per_production,
    ) = generate_problem(base_seed, target_depth)

    run_synthetic_experiments(
        benchmark_name=f"synthetic_{seed}_{target_depth}_{fitness_functions[args.fitness][0]}",
        base_seed=base_seed,
        timeout=args.timeout,
        seed=seed,
        grammar=grammar,
        representation_index=args.representation,
        target_individual=target_individual,
        target_depth=target_depth,
        fitness_function=fitness_functions[args.fitness],
        non_terminals_count=non_terminals_count,
        recursive_non_terminals_count=recursive_non_terminals_count,
        average_productions_per_terminal=average_productions_per_terminal,
        non_terminals_per_production=non_terminals_per_production,
    )
