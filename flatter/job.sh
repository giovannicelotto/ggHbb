#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                       # 2G for Data needed   
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=1:00:00  
#SBATCH --output=/t3home/gcelotto/slurm/output/flat/%x.out    # Alternative method using job name and job ID
#SBATCH --error=/t3home/gcelotto/slurm/output/flat/%x.out 
#SBATCH --dependency=singleton

echo "Before getting arguments"
source_dir="/scratch"
nanoFileName="$1"
maxEntries="$2"
maxJet="$3"
isMC="$4"
process="$5"
fileNumber="$6" 
flatPath="$7"
method="$8"
jec="$9"
echo "Before startin the python"
echo "Method is "$method
echo "JEC is "$jec
python /t3home/gcelotto/ggHbb/flatter/treeFlatter.py $nanoFileName $maxEntries $maxJet $isMC $process $method $jec


echo "Going to copy"
echo "From "$source_dir"/"$process"_"$fileNumber".parquet"
echo "To root://t3dcachedb.psi.ch:1094//"$flatPath

echo "Analyze scratch"
cd /scratch
pwd
ls
xrdcp -f -N $source_dir/$process"_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$flatPath