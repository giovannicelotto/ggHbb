#!/bin/bash
#SBATCH --output=/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/%x-%j.out
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
process=$3

echo "$file_path"
python /t3home/gcelotto/ggHbb/PNN/slurm/predict.py $file_path $isMC

# extract the fileName:
filename=$(basename "$file_path")
number=$(echo "$filename" | sed 's/.*_\([0-9]\+\)\.parquet/\1/')
source_dir="/scratch"

echo "Moving to "
echo "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNN/yMC$isMC"_"fn$number".parquet"
echo "yMC$isMC"_"fn$number".parquet"
xrdcp -f -N "$source_dir/yMC$isMC"_"fn$number".parquet" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNN/$process"/"yMC$isMC"_"fn$number".parquet"
