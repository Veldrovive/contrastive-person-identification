#!/bin/bash

# This script adds the current directory to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run source ./setup_pythonpath.sh to add the current directory to the PYTHONPATH