#!/bin/bash
#SBATCH --job-name=send            # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=2G                          # Memory per node
#SBATCH --time=0-1:00:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/slurm/output/send.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/slurm/output/send.out    # Output file for stderr


pTClass=$1
echo $pTClass

# Load necessary modules
#conda activate myenv
rm /t3home/gcelotto/ggHbb/NN/NNoutputFiles/*.out
# Submit jobs for signal
echo "Data"
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC0_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 25))
        sbatch --export=file_path=$file,isMC=0,pTClass=$pTClass --job-name=pred0$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi 
done


echo "MC"
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    #echo "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC1_fn"$fileNumber".parquet"
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC1_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=1,pTClass=$pTClass --job-name=pred1$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi
done

for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC36_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=36,pTClass=$pTClass --job-name=pred36$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi
done

for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC20_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=20,pTClass=$pTClass --job-name=pred20$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi
done

for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC21_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=21,pTClass=$pTClass --job-name=pred21$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi

done
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC22_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=22,pTClass=$pTClass --job-name=pred22$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi

    #sbatch --export=file_path=$file,isMC=22,pTClass=$pTClass --job-name=pred22$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
done

for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf/*.parquet; do
    basefile=$(basename "$file")
    fileNumber=$(echo "$basefile" | sed -E 's/.*_([0-9]+)\.parquet/\1/')
    if [ -e "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions/yMC23_fn"$fileNumber"_pt"$pTClass".parquet" ]; then
        :
    else
        number=$((1 + RANDOM % 50))
        sbatch --export=file_path=$file,isMC=23,pTClass=$pTClass --job-name=pred23$number /t3home/gcelotto/ggHbb/NN/slurm/predict.sh
    fi
done