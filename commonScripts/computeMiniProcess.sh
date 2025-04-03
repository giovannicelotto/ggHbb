#!/bin/bash
#SBATCH --job-name=CPU_job            # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=2G                          # Memory per node
#SBATCH --time=0-1:00:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/slurm/output/job.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/job.out    # Output file for stderr

pN="$1"

python /t3home/gcelotto/ggHbb/commonScripts/computeMini.py -pN $pN
