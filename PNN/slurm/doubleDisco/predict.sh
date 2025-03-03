#!/bin/bash
#SBATCH --output=/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=0-00:10:00
#SBATCH --dependency=singleton
#. /t3home/gcelotto/.bashrc
#conda activate myenv

# Run the prediction script
#echo "THis is the file_path"
file_path=$1
pN=$2
process=$3
modelName=$4
multigpu=$5

echo "$file_path"
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/predictTorch.py $file_path $process $modelName $multigpu

# extract the fileName:
filename=$(basename "$file_path")
number=$(echo "$filename" | sed 's/.*_\([0-9]\+\)\.parquet/\1/')
source_dir="/scratch"

echo "Moving to "
echo "yMC"$pN"_fn"$number".parquet"

xrdcp -f -N "$source_dir/yDD_"$process"_FN"$number".parquet" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_"$modelName"/"$process"/yDD_${process}_FN${number}.parquet"
xrdfs root://t3dcachedb.psi.ch stat /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_$modelName/$process/yDD_${process}_FN${number}.parquet
