#!/bin/bash
git pull 
pyenv local 3.11.1
source venv/bin/activate
python -m pip install -r requirements.txt