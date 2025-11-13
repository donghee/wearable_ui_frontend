#!/bin/bash

cd "$(dirname "$0")/.."

pkill sbcl

# Run background with logging
echo "Starting ACT-R ..."
sbcl --load "quicklisp/setup.lisp" --load "wde/usability/extend/load-act-r.lisp" --load "wde/usability/extend/usability/system_interface.lisp" --eval "(usability-system-interface:run-server)" --quit &

sleep 7 

# Run frontend service
echo "Starting frontend..."
. .venv/bin/activate
python3 main.py

pkill sbcl
