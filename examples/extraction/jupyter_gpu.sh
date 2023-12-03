#!/bin/bash
#SBATCH -c 1
#SBATCH --job-name=jupyter-kernel
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32GB
#SBATCH -o ./jupyter.log

# cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
