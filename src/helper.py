from importlib import import_module
from os import listdir
from os.path import isfile, join
import shutil
import pandas as pd
import os 

def get_eval_method(example_path, method_name):
    print(example_path)
    example_path = example_path.replace('/', '.')
    mod = import_module(example_path[:-3], 'geneticengine')
    return getattr(mod, method_name)

def create_folder(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

