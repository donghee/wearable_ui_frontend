#!/bin/bash

cd "$(dirname "$0")/.."

unzip quicklisp.zip -y
unzip wde.zip -y

# Delete all .fasl files for different sbcl versions
find . -name "*.fasl" -type f -delete

# Delete .venv directory if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
fi

# Install sbcl if not installed
sudo apt-get install -y sbcl

# Setup for frontend Python environment
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
