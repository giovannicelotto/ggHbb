#!/bin/bash
#SBATCH --output=/t3home/gcelotto/ggHbb/NN/NNoutputFiles/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/NN/NNoutputFiles/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:20:00
#SBATCH --dependency=singleton
#. /t3home/gcelotto/.bashrc
#conda activate myenv

# Run the prediction script
echo "THis is the file_path"
file_path=$1
isMC=$2
pTClass=$3
echo "$file_path"
python /t3home/gcelotto/ggHbb/NN/slurm/predict.py $file_path $isMC $pTClass

# extract the fileName:
filename=$(basename "$file_path")
number=$(echo "$filename" | sed 's/.*_\([0-9]\+\)\.parquet/\1/')
source_dir="/scratch"

echo "Moving to "
echo "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC$isMC"_"fn$number"_pt$pTClass".parquet"
echo "yMC$isMC"_"fn$number"_pt$pTClass".parquet"
xrdcp -f -N "$source_dir/yMC$isMC"_"fn$number"_pt$pTClass".parquet" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC$isMC"_"fn$number"_pt$pTClass".parquet"

