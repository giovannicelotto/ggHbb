#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                       # 2G for Data needed   
#SBATCH --partition=standard              # Specify your cluster partition
#SBATCH --time=12:00:00  
#SBATCH --output=/t3home/gcelotto/slurm/output/saveFlat_1.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/saveFlat_1.out    # Output file for stderr
#SBATCH --dependency=singleton

source_dir="/scratch"
nanoFileName="$1"
maxEntries="$2"
isMC="$3"
process="$4"
fileNumber="$5" 
flatPath="$6"
python /t3home/gcelotto/ggHbb/flatter/treeFlatterNew.py $nanoFileName $maxEntries $isMC $process # to pass all the arguments
pwd
xrdcp -f -N $source_dir/$process"_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$flatPath