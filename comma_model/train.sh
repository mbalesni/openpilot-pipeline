#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -J comma_gt
#SBATCH -o ./logtrainfm-%j.out # STDOUT

#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:tesla:5
#SBATCH --exclude=falcon4

python train.py --datatype "gen_gt" --phase "train" --batch_size 1 --modeltype "onnx"