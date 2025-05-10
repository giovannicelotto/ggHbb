modelName=$1
boosted=$2
for i in {0,1,3,4,19,20,21,22,36,37,43}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -b $boosted
done
for i in {44,45,46,47,48,49,50,51,53,54,55}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -b $boosted
done
for i in {56,57,58,59,60,61,62,63,65,66,67}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -b $boosted
done

#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 35 -n -1 -m $modelName -b $boosted
#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 36 -n -1 -m $modelName -b $boosted
#python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN 43 -n -1 -m $modelName -b $boosted



# JEC
for i in {0..600}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -b $boosted -JEC 1
done



# DATA


# 1A 2A 3A
#for i in {0..18}; do
#  python /t3home/gcelotto/ggHbb/PNN/slurm/mjj/runPredictions.py -MC 0 -pN $i -n -1 -m $modelName -b $boosted
#done
