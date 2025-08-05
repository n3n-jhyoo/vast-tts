#!/bin/bash

set -e -o pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
SERVER_DIR="$WORKSPACE_DIR/model-server"
ENV_PATH="$SERVER_DIR/.venv"
LOG_DIR="$WORKSPACE_DIR/logs"
DEBUG_LOG="$LOG_DIR/debug_model.log"
MODEL_LOG="$LOG_DIR/model.log"

mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

mkdir -p "$LOG_DIR"
exec &> >(tee -a "$DEBUG_LOG")

echo "start_model_server.sh"
date

echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "SERVER_DIR: $SERVER_DIR"
echo "ENV_PATH: $ENV_PATH"
echo "DEBUG_LOG: $DEBUG_LOG"
echo "MODEL_LOG: $MODEL_LOG"

if [ ! -d "$ENV_PATH" ]; then
    echo "setting up venv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.local/bin/env
    git clone https://github.com/n3n-jhyoo/vast-tts.git "$SERVER_DIR"

    uv venv --managed-python "$ENV_PATH" -p 3.10
    source "$ENV_PATH/bin/activate"
    uv pip install -r "$SERVER_DIR/requirements.txt"
else
    source ~/.local/bin/env
    source "$ENV_PATH/bin/activate"
    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"
fi


cd "$SERVER_DIR"

(python main.py |& tee -a "$MODEL_LOG") &
echo " launching model server done"