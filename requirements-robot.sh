#!/bin/bash

# Create the virtual environment
python -m venv venv
# Activate the venv environment
.\venv\Scripts\activate
# Install the dependencies to the robot environment
pip install -r requirements-robot.txt