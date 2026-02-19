#!/bin/bash
#SBATCH --job-name=saveDFS              # JobName
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --mem=128G                       # Memory per node
#SBATCH --time=0:20:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=standard            # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/ggHbb/flatter/saveDFS_p.out
#SBATCH --error=/t3home/gcelotto/ggHbb/flatter/saveDFS_p.err

period=$1
#/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/PNN/NN_highPt/classifier_TorchGPU_HighPt.py -l $lambda_disco -d 0 -e 250 -eS 1000 -b 3 -lr 0.001 -bs 8192 -n 32,16 
#/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/PNN/NN_highPt/classifier_TorchGPU_highPt_PlotOnly.py -v $lambda_disco -dt Jan14 -b 3 -s 0

/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/flatter/savedfs.py --runData 1 --period $period 
xrdcp -f -N "/scratch/dataframes_Data"$period"_Jan21_3_50p0.parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/"
xrdcp -f -N "/scratch/lumi_Data"$period"_Jan21_3_50p0.npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/lumi_Data"$period"_Jan21_3_50p0.npy"
