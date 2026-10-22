#!/bin/bash

EXEC=./coloring_test

echo "Running Nsight Compute..."

ncu \
  --metrics gpu__time_duration.sum \
  --csv \
  --page raw \
  --log-file profile.csv \
  "$EXEC"

echo
echo "===== Parsed Profile ====="
python3 parse_profile.py