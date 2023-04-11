#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi


if [[ $OSTYPE == 'darwin'* ]]; then
    PYTHONVERSION="3.10.9"
else
    PYTHONVERSION="3.11.1"
fi

pyenv install --skip-existing $PYTHONVERSION
pyenv local $PYTHONVERSION
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
