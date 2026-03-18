cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti7.txt \
                datacardMulti8.txt  > combined_datacard.txt
cd /t3home/gcelotto/ggHbb/WSFit/datacards/Zbb_Fit
N_POINTS=100
combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/combined_datacard.txt \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.,2. \
  --freezeParameters r \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --saveNLL \
  --algo grid --points $N_POINTS \
  -n rateZbbScan_combined

combine -M MultiDimFit /t3home/gcelotto/ggHbb/WSFit/datacards/combined_datacard.txt \
  -t -1 --expectSignal 1.0 \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.,2 \
  --freezeParameters r \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --saveNLL \
  --algo grid --points $N_POINTS \
  -n  rateZbbScan_exp_combined

plot1DScan.py \
  higgsCombinerateZbbScan_combined.MultiDimFit.mH120.root \
  --POI rateZbb \
  --main-label "Observed" \
  --others "higgsCombinerateZbbScan_exp_combined.MultiDimFit.mH120.root:Expected:2" \
  -o  scan_rateZbb_obs_exp_combined

