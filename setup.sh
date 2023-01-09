#!/bin/bash
pyenv install 3.11.1
pyenv local 3.11.1
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

mkdir Gengy_implementation/results
# mkdir archive
# mkdir logs
