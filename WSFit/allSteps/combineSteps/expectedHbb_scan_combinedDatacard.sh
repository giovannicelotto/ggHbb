cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti7.txt \
                datacardMulti8.txt  > combined_datacard.txt

N_POINTS=40


combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/combined_datacard.txt \
  -t -1 --expectSignal 1.0 \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.5,1.5 \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --rMin=-5 --rMax=7 \
  --saveNLL \
  --algo grid --points $N_POINTS \
  -n  rateHbbScan_exp_combined

combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/combined_datacard.txt \
  -t -1 --expectSignal 1.0 \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.5,1.5 \
  --freezeParameters rateZbb \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --saveNLL \
  --algo grid --points $N_POINTS \
  -n  rateHbbScan_exp_combined_ZbbFrozen

plot1DScan.py \
  higgsCombinerateHbbScan_exp_combined.MultiDimFit.mH120.root \
  --main-label "Expected (Zbb Free)" \
  --others "higgsCombinerateHbbScan_exp_combined_ZbbFrozen.MultiDimFit.mH120.root:Expected (Zbb Frozen):2" \
  -o  scan_rateHbb_obs_exp_combined

