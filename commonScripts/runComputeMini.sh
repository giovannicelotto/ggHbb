for i in {0..22}; do
  sbatch /t3home/gcelotto/ggHbb/commonScripts/computeMiniProcess.sh $i
done

sbatch /t3home/gcelotto/ggHbb/commonScripts/computeMiniProcess.sh 35
sbatch /t3home/gcelotto/ggHbb/commonScripts/computeMiniProcess.sh 36
sbatch /t3home/gcelotto/ggHbb/commonScripts/computeMiniProcess.sh 37