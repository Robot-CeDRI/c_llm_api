#!/bin/bash

# Create the virtual environment
python3 -m venv venv
# Activate the venv environment
source ./venv/bin/activate
# Install the dependencies to the robot environment
pip install -r requirements-robot.txt