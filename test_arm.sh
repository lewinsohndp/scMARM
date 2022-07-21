#!/bin/bash
#SBATCH --mem=32g
#SBATCH --array=2,5,10,15,20,30,40,50
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=log-%j.out
#SBATCH --error=log-%j.err
#SBATCH --gres=gpu:1

python -u test_arm.py /home/groups/ConradLab/daniel/0.4_marm_test/output_${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_ID}
