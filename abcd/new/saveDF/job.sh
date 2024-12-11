#!/bin/bash
#SBATCH --job-name=saveDFs            # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=128G                          # Memory per node
#SBATCH --time=0-2:00:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=standard              # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/ggHbb/abcd/new/saveDF/OutErr.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/ggHbb/abcd/new/saveDF/OutErr.out    # Output file for stderr


python /t3home/gcelotto/ggHbb/abcd/new/saveDF/saveDFs.py
source_dir="/scratch"
xrdcp -f -N $source_dir/dataframes.pkl root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes.pkl