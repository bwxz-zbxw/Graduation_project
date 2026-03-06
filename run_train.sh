#!/bin/bash
#JSUB -q gpu
#JSUB -n 4
#JSUB -gpgpu 1
#JSUB -o c:/Users/ASUS/Desktop/Graduation_project/train_output.log
#JSUB -e c:/Users/ASUS/Desktop/Graduation_project/train_error.log
#JSUB -J sunrgbd_train

# 1. Load Modules
module load cuda/11.6
module load python/anaconda3

# 2. Activate Environment
# Method 1: If using pre-installed conda
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate my_grad_env

# 3. Navigate to Project Directory
# Assuming the code is uploaded to /home/your_username/Graduation_project
cd /home/$USER/Graduation_project

# 4. Run Training
echo "Starting training..."
python train.py
