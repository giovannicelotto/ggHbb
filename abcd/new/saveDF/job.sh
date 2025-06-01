#!/bin/bash
#SBATCH --job-name=saveDFs            # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=128G                          # Memory per node
#SBATCH --time=0-2:00:00                 # Maximum runtime (D-HH:MM:SS)
#SBATCH --partition=standard              # Specify your cluster partition
#SBATCH --output=/t3home/gcelotto/ggHbb/abcd/new/saveDF/OutErr.out  # Output file for stdout
#SBATCH --error=/t3home/gcelotto/ggHbb/abcd/new/saveDF/OutErr.out    # Output file for stderr

model=$1
MC=$2
pN=$3


python /t3home/gcelotto/ggHbb/abcd/new/saveDF/saveDfs_DoubleDisco_multiArgument.py -MC $MC -pN $pN -s 1 -m $model
unique_id=$pN
while read -r filepath; do
    xrdcp -f -N "$filepath" root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/$model/$(basename "$filepath")
done < /scratch/output_path_${unique_id}.txt


#for i in {1..18}; do   sbatch /t3home/gcelotto/ggHbb/abcd/new/saveDF/job.sh Apr01_1000p0 0 $i; done