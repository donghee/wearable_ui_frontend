#!/bin/bash

cd "$(dirname "$0")/.."

if [ -f /tmp/wearable_ui_frontend.pid ]; then
    echo "Frontend is already running. To restart, please stop it first."
    kill -9 `cat /tmp/wearable_ui_frontend.pid`
fi

# Run wearable usability service
# Check if ACT-R exists and kill it
pkill -9 $(pgrep sbcl)

# Run ACT-R in tmux
echo "Starting ACT-R in tmux session..."
tmux new-session -d -s wearable_ui_frontend -f /dev/null
tmux send-keys -t wearable_ui_frontend C-m
sleep 2
tmux send-keys -t wearable_ui_frontend 'sbcl --load "quicklisp/setup.lisp" --load "wde/usability/extend/load-act-r.lisp" --load "wde/usability/extend/usability/system_interface.lisp"' C-m

echo "Wait to run ACT-R..."

sleep 7 

# Run wearable ui frontend service 
echo "Starting frontend..."
. .venv/bin/activate
python3 main.py &
PID_FRONTEND=$!
echo $PID_FRONTEND > /tmp/wearable_ui_frontend.pid
wait $PID_FRONTEND

