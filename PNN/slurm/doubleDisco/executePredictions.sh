modelName=$1 #Apr01_1000p0
epoch=$2 #1050
#python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN 7 -n -1 -m $modelName -gpu 1 -e $epoch
for i in {0,1,3,4,19,20,21,22,35,36,43}; do
  python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN $i -n -1 -m $modelName -gpu 1 -e $epoch
done
#####VBF and Z100to200
#python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 36 -n -1 -m $modelName -gpu 1 -e $epoch
#python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 35 -n -1 -m $modelName -gpu 1 -e $epoch
#python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 14 -n -1 -m $modelName -gpu 1 -e $epoch
#python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 1 -pN 37 -n -1 -m $modelName -gpu 1 -e $epoch
##
## 1A 2A 3A
#for i in {2,1}; do 
#  python /t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/runPredictions.py -MC 0 -pN $i -n -1 -m $modelName -gpu 1 -e $epoch
#done
