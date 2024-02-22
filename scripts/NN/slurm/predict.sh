#!/bin/bash
#SBATCH --job-name=predict_job
#SBATCH --output=/t3home/gcelotto/ggHbb/scripts/NN/slurm/slurmOutput/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/scripts/NN/slurm/slurmOutput/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
. /t3home/gcelotto/.bashrc
conda activate myenv

# Run the prediction script
python predict.py $file_path $isMC
