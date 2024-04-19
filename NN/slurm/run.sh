#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=/t3home/gcelotto/ggHbb/NN/NNoutputFiles/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/NN/NNoutputFiles/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=0-01:00:00


pTClass=$1
echo $pTClass

# Load necessary modules
#conda activate myenv
rm /t3home/gcelotto/ggHbb/NN/NNoutputFiles/*.out
# Submit jobs for signal
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**/*.parquet; do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=0,pTClass=$pTClass --job-name=pred0$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others/*.parquet; do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=1,pTClass=$pTClass --job-name=pred1$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400/*.parquet
do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=20,pTClass=$pTClass --job-name=pred20$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600/*.parquet
do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=21,pTClass=$pTClass --job-name=pred21$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800/*.parquet
do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=22,pTClass=$pTClass --job-name=pred22$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf/*.parquet
do
    number=$((1 + RANDOM % 25))
    sbatch --export=file_path=$file,isMC=23,pTClass=$pTClass --job-name=pred23$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done