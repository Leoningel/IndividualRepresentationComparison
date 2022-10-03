#!/bin/bash
cd GeneticEngine 
git pull 
python3 -m pip install -r requirements.txt
cd ..
python3 -m pip install -r requirements.txt