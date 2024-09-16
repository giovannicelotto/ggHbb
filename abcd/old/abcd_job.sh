#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=1:00:00
#SBATCH --output=/t3home/gcelotto/slurm/output/job.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/job.out    # Output file for stderr
#SBATCH --dependency=singleton

python /t3home/gcelotto/ggHbb/bkgEstimation/scripts/process_file.py $@  # to pass all the arguments