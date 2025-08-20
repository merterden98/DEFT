#!/bin/bash
#SBATCH --mem=64000mb             # memory
#SBATCH --cpus-per-task=4
#SBATCH --partition=cellbio-dgx
#SBATCH --job-name="tunnel"
#SBATCH --time=24:00:00     # walltime
#SBATCH --output=./runs/run_%j.log  # stores output
#SBATCH --error=./runs/run_%j.log   # stores error messages

conda activate SaProt2
python3 get_jay_aln.py