#!/bin/bash
#SBATCH --account=gpu_gres                                  # to access gpu resources
#SBATCH --partition=gpu                                     # Specify your cluster partition
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10000
#SBATCH --gres=gpu:4                                       # request  N GPU
#SBATCH --time=0-10:00:00                                   # Maximum runtime (D-HH:MM:SS)
#SBATCH --output=/t3home/gcelotto/slurm/output/DoubleD_GPU_hp.out    # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/DoubleD_GPU_hp.out     # Output file for stderr

#export PATH=/usr/local/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

conda init
conda activate pytorch_test

lambda_disco="$1"
bs="$2"
lr="$3"
nNodes="$4"
modelName="$5"
echo "lambda " 
echo $lambda_disco
#s="$2"
# 16384
# 32768
# /t3home/gcelotto/ggHbb/PNN/DoubleDisco_MultiGPU.py
# /t3home/gcelotto/ggHbb/PNN/DoubleDiscoPNNTorchGPU.py


export MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using MASTER_PORT=$MASTER_PORT"


/work/gcelotto/miniconda3/envs/pytorch_test/bin/python /t3home/gcelotto/ggHbb/PNN/DoubleDisco_MultiGPU.py -l $lambda_disco -e 1000 -bs $bs -lr $lr -n "$nNodes" -save 0 -modelName $modelName

xrdcp -f -N "/scratch/nn1_"$modelName".pth" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dd_models/"
xrdcp -f -N "/scratch/nn2_"$modelName".pth" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dd_models/"
xrdcp -f -N "/scratch/sum_"$modelName".txt" "root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dd_models/"
