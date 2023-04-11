#!/bin/bash

# Your list of test functions
test_data=("iris" "breast_cancer" "digits" "wine")

session_name="test_data"

# Create the first window with the first test function
tmux new-session -d -s "${session_name}" -n "${test_data[0]}" "bash -c 'source ~/.zshrc; conda activate hebo; bash run_toy.sh \"${test_data[0]}\"; bash'"

# Create the remaining windows with the remaining test functions
for test_dt in "${test_data[@]:1}"; do
  tmux new-window -t "${session_name}" -n "${test_dt}" "bash -c 'source ~/.zshrc; conda activate hebo; bash run_toy.sh \"${test_dt}\"; bash'"

done

# Attach to the created session
tmux attach-session -t "${session_name}"
