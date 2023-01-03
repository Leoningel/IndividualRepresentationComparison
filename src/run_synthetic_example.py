
from argparse import ArgumentParser
from examples.utils.wrapper import run_experiments

from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.grammatical_evolution.ge import ge_representation
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import dsge_representation

not_complex = { 
    'n_class_abc': 1,
    'n_class_0_ch': 3,
    'n_class_2_ch': 2,
    'max_var_per_class': 2,
}

complex = { 
    'n_class_abc': 2,
    'n_class_0_ch': 5,
    'n_class_2_ch': 3,
    'max_var_per_class': 4,
}

very_complex = { 
    'n_class_abc': 4,
    'n_class_0_ch': 7,
    'n_class_2_ch': 5,
    'max_var_per_class': 6,
}

if __name__ == "__main__":
    representations = [ 'ge', 'dsge', 'treebased' ]
    
    parser = ArgumentParser()
    parser.add_argument("-e", "--example", dest="example", type=str)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=123)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()
    representation = representations[args.representation]
    seed = args.seed

    example = args.example
    if example == 'easy_not_complex':
        level_of_hardens = 'easy'
        grammar_specs = not_complex
    elif example == 'easy_complex':
        level_of_hardens = 'easy'
        grammar_specs = complex
    elif example == 'easy_very_complex':
        level_of_hardens = 'easy'
        grammar_specs = very_complex
    elif example == 'medium_not_complex':
        level_of_hardens = 'medium'
        grammar_specs = not_complex
    elif example == 'medium_complex':
        level_of_hardens = 'medium'
        grammar_specs = complex
    elif example == 'medium_very_complex':
        level_of_hardens = 'medium'
        grammar_specs = very_complex
    elif example == 'hard_not_complex':
        level_of_hardens = 'hard'
        grammar_specs = not_complex
    elif example == 'hard_complex':
        level_of_hardens = 'hard'
        grammar_specs = complex
    elif example == 'hard_very_complex':
        level_of_hardens = 'hard'
        grammar_specs = very_complex
    else:
        raise Exception(f"The example {example} is not included. Included examples: easy_not_complex, easy_complex, easy_very_complex, medium_not_complex, medium_complex, medium_very_complex, hard_not_complex, hard_complex, hard_very_complex")
    from examples.dynamic_grammar_ex import grammar_and_ff_def, params
    n_class_abc = grammar_specs['n_class_abc']
    n_class_0_ch = grammar_specs['n_class_0_ch']
    n_class_2_ch = grammar_specs['n_class_2_ch']
    max_var_per_class = grammar_specs['max_var_per_class']
    grammar, ff_train = grammar_and_ff_def(g_seed=seed, n_class_abc=n_class_abc, n_class_0_ch=n_class_0_ch, n_class_2_ch=n_class_2_ch, max_var_per_class=max_var_per_class, level_of_hardness=level_of_hardens)

    run_experiments(grammar, ff=ff_train(), ff_test=None, folder_name=f"synthetic/{example}", seed=seed, params=params, representation=representation)
