#!/bin/bash
#SBATCH --job-name=saveDFS              # JobName
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --mem=128G                       # Memory per node
#SBATCH --time=0:20:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=standard            # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/ggHbb/flatter/saveDFS_all.out
#SBATCH --error=/t3home/gcelotto/ggHbb/flatter/saveDFS_all.err

modelName="Mar31_11_50p0"
#/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/PNN/NN_highPt/classifier_TorchGPU_HighPt.py -l $lambda_disco -d 0 -e 250 -eS 1000 -b 3 -lr 0.001 -bs 8192 -n 32,16 
#/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/PNN/NN_highPt/classifier_TorchGPU_highPt_PlotOnly.py -v $lambda_disco -dt Jan14 -b 3 -s 0

/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/flatter/savedfs.py --runData 1 
#xrdcp -f -N "/scratch/dataframes_Data"$period"_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
#xrdcp -f -N "/scratch/lumi_Data"$period"_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data"$period"_"$modelName".npy"
##
xrdcp -f -N "/scratch/dataframes_Data1A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data2A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data3A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data4A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data5A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data6A_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
##
##
xrdcp -f -N "/scratch/dataframes_Data1B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data2B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data3B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data4B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data5B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data6B_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
##
xrdcp -f -N "/scratch/dataframes_Data1C_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data2C_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data3C_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data4C_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data5C_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
##
##
xrdcp -f -N "/scratch/dataframes_Data1D_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data2D_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data3D_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data4D_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
xrdcp -f -N "/scratch/dataframes_Data5D_"$modelName".parquet" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/"
##
xrdcp -f -N "/scratch/lumi_Data1A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data1A_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data2A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data2A_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data3A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data3A_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data4A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data4A_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data5A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data5A_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data6A_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data6A_"$modelName".npy"
##
xrdcp -f -N "/scratch/lumi_Data1B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data1B_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data2B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data2B_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data3B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data3B_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data4B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data4B_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data5B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data5B_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data6B_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data6B_"$modelName".npy"
##
xrdcp -f -N "/scratch/lumi_Data1C_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data1C_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data2C_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data2C_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data3C_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data3C_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data4C_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data4C_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data5C_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data5C_"$modelName".npy"
##
xrdcp -f -N "/scratch/lumi_Data1D_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data1D_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data2D_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data2D_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data3D_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data3D_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data4D_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data4D_"$modelName".npy"
xrdcp -f -N "/scratch/lumi_Data5D_"$modelName".npy" "root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"$modelName"/lumi_Data5D_"$modelName".npy"
#