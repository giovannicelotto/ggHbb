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
fileName=$1
prefix=$2
outFolder=$3
fileNumber=$4
python /t3home/gcelotto/ggHbb/BDT/dataPreparing/genTreeFlatter.py $fileName $prefix $fileNumber
pwd
xrdcp -f -N $source_dir/$prefix"_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$outFolder