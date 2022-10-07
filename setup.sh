#!/bin/bash
# git clone -b RepresentationComparison https://github.com/alcides/GeneticEngine.git
cd GeneticEngine 
python -m pip install -r requirements.txt
cd ..
python -m pip install -e GeneticEngine
python -m pip install -r requirements.txt
# mkdir results
# mkdir archive
