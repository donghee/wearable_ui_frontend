#!/bin/bash

cd "$(dirname "$0")/.."

sbcl --load "quicklisp/setup.lisp" --load "wde/usability/extend/load-act-r.lisp" --load
"wde/usability/extend/usability/system_interface.lisp"

#python3 -m venv .venv
#source .venv/bin/activate
#pip3 install -r requirements.txt
#python3 main.py
