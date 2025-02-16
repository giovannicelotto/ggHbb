modelName=$1
for i in {0..22}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN $i -n 100 -m $modelName
done
#VBF and Z100to200
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 36 -n -1 -m $modelName
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 35 -n -1 -m $modelName

# 1A 2A 3A
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN 0 -n -1 -m $modelName
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN 1 -n -1 -m $modelName
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN 2 -n -1 -m $modelName
python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN 3 -n -1 -m $modelName