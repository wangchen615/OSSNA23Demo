#!/bin/bash

# Set the virtual environment path
venv_path="/home/chenw/venvs/bloom-load-generator"

# Activate the virtual environment
source "$venv_path/bin/activate"

# Define the energy levels
declare -a energy_levels=("100W" "150W" "200W" "250W")

# Define the script path
script_path="bloom-generate-query.py"

# Check if the script file exists
if [ ! -f "$script_path" ]; then
    echo "Error: $script_path not found."
    exit 1
fi

# Iterate through the energy levels
for energy_level in "${energy_levels[@]}"; do
    echo "Setting GPU energy conservation level to ${energy_level}W"
    sudo nvidia-smi -pl "$energy_level"

    # Run the Python script with the current energy_level
    exp_name="exp-${energy_level}"
    echo "Running script with --exp-name=$exp_name"
    python3 "$script_path" --host localhost --port 5001 --exp-name "$exp_name" --metric-endpoint http://localhost:9090
done

echo "Energy Conservation Experiment Execution Completed."

# Deactivate the virtual environment
deactivate
