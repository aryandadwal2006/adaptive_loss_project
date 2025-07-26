#!/usr/bin/env bash
CONFIG=$1
source venv/Scripts/Activate.ps1
python src/main.py ablate --config "$CONFIG"
