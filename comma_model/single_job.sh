#!/bin/bash
 
#SBATCH -J comma_train
#SBATCH -o ./logs/log-%j.out # STDOUT
#SBATCH -t 09:59:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2

echo "this script will launch create_files.py with every dir path that i want to edit"
python create_files.py $1 $2

