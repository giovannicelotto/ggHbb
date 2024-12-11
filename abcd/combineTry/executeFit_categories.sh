cd datacards/
combineCards.py ch1=shapeZdatacard_lcPass.txt ch2=shapeZdatacard_lcFail_b2Med.txt ch3=shapeZdatacard_lcFail_b2Tight.txt > combined_datacard.txt
cd ..
datacard="datacards/combined_datacard.txt"
# Fit observed 
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin -0 --rMax 2 --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
# Fit expected
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin -0 --rMax 2 --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
# Plot likelihood scan for expected and observed
plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90_observed.root --output scan_categories_exp_obs --others "/t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed"

text2workspace.py $datacard -m 90
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d datacards/combined_datacard.root  -m 90 -o impacts_categories.json
plotImpacts.py -i impacts_categories.json -o impacts_categories
mv impacts_categories.p* plots/


# Run a fit with all nuisance parameters floating and store the workspace in an output file
combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/combined_datacard.root -M MultiDimFit --saveWorkspace -n zbb_cat.postfit -m 90 --points 100
# Run a scan from the postfit workspace
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit -n zbb_cat.total --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,2 -m 90 --points 100
# Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,2  --freezeNuisanceGroups theory -n zbb.freeze_theory -m 90 --points 100
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,2  --freezeParameters allConstrainedNuisances -n zbb_cat.freeze_all -m 90

plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.total.MultiDimFit.mH90.root --main-label "Total Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb_cat.freeze_all.MultiDimFit.mH90.root:"Stat only":2  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat"  -o breakdown_cat
mv breakdown_cat.p* plots/