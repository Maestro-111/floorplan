#!/bin/bash

# set config file
# DEMO RUN help: ./run.sh -h
# DEMO RUN with param: ./run.sh -f ./config/config2.ini
# PROD RUN: ./run.sh -f ./config/config.ini

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
export RMBASE_FILE_PYTHON=$SCRIPTPATH/config/config.ini

export DEBUG=True

# active python virtualenv if needed
# source $HOME/py3/bin/activate

# run python script
# | tee $SCRIPTPATH/log/run.log
python $SCRIPTPATH/src/main.py "$@" 2>&1
