cd datacards/
combineCards.py ch1=shapeZdatacard_lc1_cs0.txt ch2=shapeZdatacard_lc1_cs1.txt ch3=shapeZdatacard_lc2_cs0.txt ch4=shapeZdatacard_lc2_cs1.txt ch5=shapeZdatacard_lc3_cs0.txt ch6=shapeZdatacard_lc3_cs1.txt> combined_datacard.txt
cd ..
combine -M MultiDimFit datacards/combined_datacard.txt --algo grid --points 100 --rMin -0 --rMax 2 --mass 90
python3 /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90.root -o scan_categories
datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/combined_datacard.txt"

text2workspace.py $datacard -m 90
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 -o impacts_categories.json
plotImpacts.py -i impacts_categories.json -o impacts_categories


# Run a fit with all nuisance parameters floating and store the workspace in an output file
combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/combined_datacard.root -M MultiDimFit --saveWorkspace -n zbb_cat.postfit -m 90 --points 100
# Run a scan from the postfit workspace
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit -n zbb_cat.total --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4 -m 90 --points 100
# Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeNuisanceGroups theory -n zbb.freeze_theory -m 90 --points 100
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeParameters allConstrainedNuisances -n zbb_cat.freeze_all -m 90

plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.total.MultiDimFit.mH90.root --main-label "Total Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.freeze_all.MultiDimFit.mH90.root:"stat only":6  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat"  -o breakdown_cat