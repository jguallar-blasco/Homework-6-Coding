#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"

module load anaconda

# init virtual environment if needed
#conda create -n toy_classification_env python=3.7
conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies
pip install typing-extensions --upgrade
pip install --upgrade openai
python setup.py install
pip install accelerate

conda install -c pytorch faiss-gpu

# runs your code
srun python hw7_classification.py --device cuda --model "bigscience/bloomz" --batch_size "32" --lr 1e-3 --num_epochs 9
#srun python open_ai.py
