#!/bin/bash

# Declare arrays for the parameter values
num_ens_values=(1 2 3)

# Check if a command-line argument is provided
if [ "$#" -eq 1 ]; then
    dataset_values=("$1")
else
    dataset_values=('iris')
fi

# Loop through the combinations of parameters and run the Python script
for num_ens in "${num_ens_values[@]}"; do
  for dataset in "${dataset_values[@]}"; do
    echo "Running toy_run.py with --num_ens=${num_ens} and --dataset=${dataset}"
    python toy_run.py --num_ens "${num_ens}" --dataset "${dataset}"
  done
done

echo "Finished running all combinations"
