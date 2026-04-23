cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/Zbb_Fit
N_POINTS=10
dataName=$1
rMin=0
rMax=2

echo "Processing Category ${dataName}"

combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/datacardMulti${dataName}.txt \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=$rMin,$rMax \
  --freezeParameters r \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --saveNLL \
  --cminDefaultMinimizerStrategy 0 \
  --algo grid --points $N_POINTS \
  -n rateZbbScan_${dataName}

combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/datacardMulti${dataName}.txt \
  -t -1 --expectSignal 1.0 \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=$rMin,$rMax \
  --freezeParameters r \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --cminDefaultMinimizerStrategy 0 \
  --saveNLL \
  --algo grid --points $N_POINTS \
  -n  rateZbbScan_exp_${dataName}

plot1DScan.py \
  higgsCombinerateZbbScan_${dataName}.MultiDimFit.mH120.root \
  --POI rateZbb \
  --main-label "Observed" \
  --others "higgsCombinerateZbbScan_exp_${dataName}.MultiDimFit.mH120.root:Expected:2" \
  -o  scan_rateZbb_obs_exp_${dataName}

