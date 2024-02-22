#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=1:00:00
#SBATCH --output=/t3home/gcelotto/slurm/output/job.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/job.out    # Output file for stderr
#SBATCH --dependency=singleton

python /t3home/gcelotto/ggHbb/flatter/treeFlatter.py $@  # to pass all the arguments
source_dir="/scratch"
process="WW"
fileName="$1"
process="$2"
fileNumber="$3" 
flatPath="$4"

xrdcp -f -N $source_dir/$process"_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$4