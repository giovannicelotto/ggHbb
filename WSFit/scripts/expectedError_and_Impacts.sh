cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti0.txt \
                datacardMulti1.txt \
                datacardMulti2.txt \
                datacardMulti10.txt \
                datacardMulti11.txt > combined.txt

# Expected Error based on Asimov of PDF that performs the best on Data
combine -M FitDiagnostics -d combined.txt \
         -t -1 \
         --expectSignal 1 \
        -m 125 \
        --setParameterRange r=-10,7\
         --X-rtd MINIMIZER_freezeDisassociatedParams \
         --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
         --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV\
        --cminDefaultMinimizerStrategy 0\
        -v3\
        -n expectedError\

exit
combine  --run blind combined.txt -t -1 \
        --expectSignal 1 \
        -m 125 \
        --X-rtd MINIMIZER_freezeDisassociatedParams \
        --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
        --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV,rateZbb\
        --cminDefaultMinimizerStrategy 0\
        -n combinedAsymptoticLimit
combine  --run blind combined.txt -t -1 \
    --expectSignal 1 \
    -m 125 \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
    --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV\
    --cminDefaultMinimizerStrategy 0\
    -n combinedAsymptoticLimit
#

#

echo "Freezing rateZbb"
combine -M FitDiagnostics -d combined.txt \
         -t -1 \
         --expectSignal 1 \
        -m 125 \
        --setParameterRange r=-10,7\
         --X-rtd MINIMIZER_freezeDisassociatedParams \
         --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
         --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV,rateZbb\
        --cminDefaultMinimizerStrategy 0\
        -n expectedErrorRateZbbFrozen\
exit
# Observed and Expected rateZbb likelihood scan
combine -M MultiDimFit combined.txt \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.5,1.5 \
  --freezeParameters r \
    --X-rtd MINIMIZER_freezeDisassociatedParams\
    --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
    --saveNLL\
  --algo grid --points 20\
    -n rateZbbScan
combine -M MultiDimFit combined.txt \
  -t -1 --expectSignal 1.0 \
  --redefineSignalPOIs rateZbb \
  --setParameters rateZbb=1.0 \
  --setParameterRanges rateZbb=0.5,1.5 \
  --freezeParameters r \
  --X-rtd MINIMIZER_freezeDisassociatedParams \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
  --saveNLL \
  --algo grid --points 20 \
  -n rateZbbScan_exp
plot1DScan.py higgsCombinerateZbbScan.MultiDimFit.mH120.root \
  --POI rateZbb \
  --main-label "Observed" \
  --others "higgsCombinerateZbbScan_exp.MultiDimFit.mH120.root:Expected:2" \
  -o scan_rateZbb_obs_exp
#
#




#Impact plots for the r_H

text2workspace.py combined.txt -m 125
combineTool.py -M Impacts -d combined.root -m 125 -n .impacts --setParameterRanges r=-1,3 --doInitialFit --robustFit 1 --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 \
                        --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
                        --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d combined.root -m 125 -n .impacts --setParameterRanges r=-1,3 --doFits --robustFit 1  --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 \
                        --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3\
                        --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d combined.root -m 125 -n .impacts --setParameterRanges r=-1,3 -o impacts_combined.json
plotImpacts.py -i impacts_combined.json -o impacts_combined --per-page 25