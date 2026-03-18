cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti1.txt \
                datacardMulti7.txt \
                datacardMulti8.txt  > combined_datacard.txt


text2workspace.py combined_datacard.txt -m 125
combineTool.py -M Impacts -d combined_datacard.txt -m 125 -n .impacts --setParameterRanges r=-30,30 --doInitialFit --robustFit 1 --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 \
#                        --freezeParameters pdfindex_1_2016_13TeV,pdfindex_7_2016_13TeV,pdfindex_8_2016_13TeV --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d combined_datacard.txt -m 125 -n .impacts --setParameterRanges r=-30,30 --doFits --robustFit 1  --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 
#                        --freezeParameters pdfindex_1_2016_13TeV,pdfindex_7_2016_13TeV,pdfindex_8_2016_13TeV --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d combined_datacard.txt -m 125 -n .impacts --setParameterRanges r=-30,30 -o "impacts/impacts_datacardMulti"$CATEGORY".json"
plotImpacts.py -i "impacts/impacts_datacardMulti"$CATEGORY".json" -o "impacts/impacts_datacardMulti"$CATEGORY