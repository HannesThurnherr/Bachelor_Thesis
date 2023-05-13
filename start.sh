#!/bin/bash
#SBATCH --mail-user=hannes.thurnherr@gmail.com
#SBATCH --mail-type=end,fail
#SBATCH --job-name="full_run"
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30

# Your code below this line

param_store=$HOME/Bachelor_Thesis/args.txt

param_a=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

source venv/bin/activate
srun python sequential_parameter_search.py ${param_a} >> log_${param_a}.txt

