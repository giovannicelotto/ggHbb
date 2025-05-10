#!/bin/bash
#SBATCH --job-name=predictions
#SBATCH --output=predictions.out
#SBATCH --error=predictions.err
#SBATCH --time=01:00:00        # adjust time as needed
#SBATCH --partition=short        # or whatever partition you need
#SBATCH --cpus-per-task=4      # adjust based on needs
#SBATCH --mem=1G               # memory per task
#SBATCH --array=0-9            # this is the key line for {0..9}

# Load your environment if needed
# source /path/to/your/env/bin/activate
# or module load python/3.X

echo "Running for task ID: $SLURM_ARRAY_TASK_ID"

python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN $SLURM_ARRAY_TASK_ID -m Apr28_600p0 -n -1 -gpu 1 -e 900
