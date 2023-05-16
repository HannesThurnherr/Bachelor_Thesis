#!/bin/bash
#SBATCH --mail-user=hannes.thurnherr@gmail.com
#SBATCH --mail-type=end,fail
#SBATCH --job-name="full_run"
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Test: Echo statements
echo "Script started"

# Test: Print current directory and its contents
echo "Current directory:"
pwd


param_store=$HOME/Bachelor_Thesis/args.txt

param_a=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

source venv/bin/activate

# Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Test: Print python version
echo "Python version:"
python --version

# Test: Run a simple python command
echo "Test Python execution:"
python -c "print('Hello, World!')"

# Run the python script and capture its output and any errors
python sequential_parameter_search.py ${param_a} >> log_${param_a}.txt 2>&1

# Check if the python script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script failed to run"
    exit 1
fi

# Test: Echo statement
echo "Script ended"
