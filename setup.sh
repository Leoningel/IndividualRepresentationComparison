#!/bin/bash
if [[ $OSTYPE == 'darwin'* ]]; then
    PYTHONVERSION="3.10.9"
else
    PYTHONVERSION="3.11.0"
fi

pyenv install --skip-existing $PYTHONVERSION
pyenv local $PYTHONVERSION

python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
mkdir Gengy_implementation/results
# mkdir archive
# mkdir logs
