modelName=$1
boosted=$2
for i in {0,1,3,4,19,20,21,22,36,37,43}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -b $boosted
done
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 35 -n -1 -m $modelName -b $boosted
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 36 -n -1 -m $modelName -b $boosted
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 43 -n -1 -m $modelName -b $boosted

# 1A 2A 3A
#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 0 -n -1 -m $modelName 
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 1 -n -1 -m $modelName -b $boosted
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 2 -n -1 -m $modelName -b $boosted
python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 3 -n -1 -m $modelName -b $boosted
#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 4 -n -1 -m $modelName 
#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN 5 -n -1 -m $modelName 