#!/bin/bash

# Set the virtual environment path
venv_path="/home/chenw/venvs/bloom-load-generator"

# Activate the virtual environment
source "$venv_path/bin/activate"

# Define the energy levels
declare -a sm_levels=("150" "270" "390" "510" "630" "750" "870" "990" "1110" "1230" "1380")

# Define the script path
script_path="bloom-generate-query.py"

# Check if the script file exists
if [ ! -f "$script_path" ]; then
    echo "Error: $script_path not found."
    exit 1
fi

# Iterate through the energy levels
for sm_level in "${sm_levels[@]}"; do
    echo "Setting GPU graphic clock level to ${energy_level}MHz"
    sudo nvidia-smi -i 0 -ac "877,$sm_level"

    # Run the Python script with the current energy_level
    exp_name="SM-${sm_level}"
    echo "Running script with --exp-name=$exp_name"
    python3 "$script_path" --host localhost --port 5001 --exp-name "$exp_name" --metric-endpoint http://localhost:9090 --num-tests 20
done

echo "Graphics CLOCK Tuning Experiment Execution Completed."

# Deactivate the virtual environment
deactivate
