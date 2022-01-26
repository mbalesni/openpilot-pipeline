#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -J comma_gt
#SBATCH -o ./logtrainfm-%j.out # STDOUT

#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon3

srun python train.py "$@"