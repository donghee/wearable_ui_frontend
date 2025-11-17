#!/bin/bash

cd "$(dirname "$0")/.."

yes | unzip quicklisp.zip
yes | unzip wde.zip
yes | unzip dist.zip

# Delete all .fasl files for different sbcl versions
find . -name "*.fasl" -type f -delete

# Delete .venv directory if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
fi

# Install wearable usability service (ACT-R)
sudo apt-get install -y sbcl tmux

# Setup for frontend Python environment
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
