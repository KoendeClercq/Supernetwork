#!/bin/sh
#
#SBATCH --job-name=“main_job”
#SBATCH --partition=compute
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=research-ceg-tp

module load 2022r2
module load py-pip

python main.py
