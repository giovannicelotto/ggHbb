#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                       # 2G for Data needed   
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=01:00:00  
#SBATCH --output=/t3home/gcelotto/slurm/output/saveGenMatchedFlat_1.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/saveGenMatchedFlat_1.out    # Output file for stderr
#SBATCH --dependency=singleton

source_dir="/scratch"
nanoFileName="$1"
process="$2"
fileNumber="$3" 
flatPath="$4"
python /t3home/gcelotto/ggHbb/genMatching/genFlatterNu/flatterWNu_slurm.py $nanoFileName $fileNumber $process # to pass all the arguments
pwd
xrdcp -f -N $source_dir/$process"_GenMatched_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$flatPath