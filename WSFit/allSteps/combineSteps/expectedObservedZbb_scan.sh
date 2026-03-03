cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
N_POINTS=10
for dataName in \
datacardMulti0.txt \
datacardMulti1.txt \
datacardMulti2.txt \
datacardMulti4.txt \
datacardMulti5.txt \
datacardMulti6.txt \
datacardMulti10.txt \
datacardMulti11.txt
do
  echo "Processing ${dataName}"

  combine -M MultiDimFit ${dataName} \
    --redefineSignalPOIs rateZbb \
    --setParameters rateZbb=1.0 \
    --setParameterRanges rateZbb=0.5,1.5 \
    --freezeParameters r \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
    --saveNLL \
    --algo grid --points $N_POINTS \
    -n rateZbbScan_${dataName%.txt}

  combine -M MultiDimFit ${dataName} \
    -t -1 --expectSignal 1.0 \
    --redefineSignalPOIs rateZbb \
    --setParameters rateZbb=1.0 \
    --setParameterRanges rateZbb=0.5,1.5 \
    --freezeParameters r \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
    --saveNLL \
    --algo grid --points $N_POINTS \
    -n rateZbbScan_exp_${dataName%.txt}

  plot1DScan.py \
    higgsCombinerateZbbScan_${dataName%.txt}.MultiDimFit.mH120.root \
    --POI rateZbb \
    --main-label "Observed" \
    --others "higgsCombinerateZbbScan_exp_${dataName%.txt}.MultiDimFit.mH120.root:Expected:2" \
    -o scan_rateZbb_obs_exp_${dataName%.txt}

done