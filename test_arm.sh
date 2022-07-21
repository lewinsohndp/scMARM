#!/bin/bash
#SBATCH --mem=32g
#SBATCH --array=2,5,10,15
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=log-%j.out
#SBATCH --error=log-%j.err
#SBATCH --gres=gpu:1

python -u test_arm.py ${SLURM_ARRAY_TASK_ID}
