#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                       # 2G for Data needed   
#SBATCH --partition=short              # Specify your cluster partition
#SBATCH --time=0:59:00  #keep it standard for ttbar
#SBATCH --output=/t3home/gcelotto/slurm/output/flat/%x.out    # Alternative method using job name and job ID
#SBATCH --error=/t3home/gcelotto/slurm/output/flat/%x.out 
#SBATCH --dependency=singleton
start_time=$(date +%s)
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
isMC="${11}"
run="${12}"
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
echo "isMC is "$isMC
echo "run is "$run
if [ "$run" = "2" ]; then
    /work/gcelotto/miniconda3/envs/myenv/bin/python \
        /t3home/gcelotto/ggHbb/flatter/treeFlatter_dict.py \
        "$nanoFileName" "$maxEntries" "$maxJet" "$pN" \
        "$process" "$method" "$jec" "$verbose" \
        "$isMC" "$fileNumber"
elif [ "$run" = "3" ]; then
    echo "Running with treeFlatter_dict_Run3.py"
    /work/gcelotto/miniconda3/envs/myenv/bin/python \
        /t3home/gcelotto/ggHbb/flatter/treeFlatter_dict_Run3.py \
        "$nanoFileName" "$maxEntries" "$maxJet" "$pN" \
        "$process" "$method" "$jec" "$verbose" \
        "$isMC" "$fileNumber"
fi


echo "Going to copy"
echo "From "$source_dir"/"$process"_"$fileNumber".parquet"
echo "To root://t3dcachedb03.psi.ch:1094//"$flatPath

echo "Analyze scratch"
cd /scratch
pwd
ls
xrdcp -f -N $source_dir/$process"_"$fileNumber.parquet root://t3dcachedb03.psi.ch:1094//$flatPath
rm $source_dir/$process"_"$fileNumber.parquet

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Print elapsed time in hh:mm:ss
printf "Job finished in %02d:%02d:%02d (hh:mm:ss)\n" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))
