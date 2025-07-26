#!/usr/bin/env bash
# Usage: ./run_adaptive.sh adaptive_default
CONFIG=$1
source venv/Scripts/Activate.ps1
python src/main.py run --config "$CONFIG" --adaptive
