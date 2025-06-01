#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                       # 2G for Data needed   
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=1:00:00  
#SBATCH --output=/t3home/gcelotto/slurm/output/flat/%x.out    # Alternative method using job name and job ID
#SBATCH --error=/t3home/gcelotto/slurm/output/flat/%x.out 
#SBATCH --dependency=singleton

echo "Before getting arguments"
#nanoFileName, str(maxEntries), str(maxJet), str(pN), process, str(fileNumber), flatPath, str(method), str(isJEC), str(args.verbose)
source_dir="/scratch"
nanoFileName="$1"
maxEntries="$2"
maxJet="$3"
pN="$4"
process="$5"
fileNumber="$6" 
flatPath="$7"
method="$8"
jec="$9"
verbose="${10}"
echo "Before startin the python"
echo "Method is "$method
echo "nanoFileName is "$nanoFileName
echo "maxEntries is "$maxEntries
echo "maxJet is "$maxJet
echo "pN is "$pN
echo "process is "$process
echo "fileNumber is "$fileNumber
echo "flatPath is "$flatPath
echo "method is "$method
echo "jec is "$jec
echo "verbose is "$verbose
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/flatter/treeFlatter_dict.py $nanoFileName $maxEntries $maxJet $pN $process $method $jec $verbose


echo "Going to copy"
echo "From "$source_dir"/"$process"_"$fileNumber".parquet"
echo "To root://t3dcachedb.psi.ch:1094//"$flatPath

echo "Analyze scratch"
cd /scratch
pwd
ls
xrdcp -f -N $source_dir/$process"_"$fileNumber.parquet root://t3dcachedb.psi.ch:1094//$flatPath