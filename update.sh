#!/bin/bash
cd GeneticEngine 
git pull 
python -m pip install -r requirements.txt
cd ..
python -m pip install -r requirements.txt