#!/bin/bash


#SBATCH -J comma_train
#SBATCH -o ./log-%j.out # STDOUT
#SBATCH -t 09:59:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20

python hevc_jpg.py