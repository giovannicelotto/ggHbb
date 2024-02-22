#!/bin/bash
#SBATCH --job-name=predictions
#SBATCH --output=/t3home/gcelotto/ggHbb/scripts/NN/slurm/slurmOutput/%x-%j.out
#SBATCH --error=/t3home/gcelotto/ggHbb/scripts/NN/slurm/slurmOutput/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00

# Load necessary modules
conda activate myenv
rm /t3home/gcelotto/ggHbb/scripts/NN/slurm/slurmOutput/*.out

# Submit jobs for signal
for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/training/*.parquet; do
    sbatch --export=file_path=$file,isMC=1 predict.sh
# Submit jobs for bkg use all bkg in others in order to have a consistent measurement of the lumi
#for file in /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot/others*.parquet; do
#    sbatch --export=file_path=$file,isMC=0 predict_job.sh
done
