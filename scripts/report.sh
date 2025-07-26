#!/usr/bin/env bash
EXP=$1  # e.g. experiments/results/adaptive_default
source venv/Scripts/Activate.ps1
python src/main.py report --exp_dir "$EXP"
