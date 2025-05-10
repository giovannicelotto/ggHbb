#!/bin/bash
cd /t3home/gcelotto/ggHbb/abcd/combineTry
datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total_reduced.txt"


#model="Apr01_1000p0"

# Use sed to replace the line in place
#sed -i "s|shapes \*    total  .*root|shapes *    total  ${new_root_file} \$PROCESS|" "$datacard"



combine -M MultiDimFit $datacard --algo grid --points 100 --rMin 0 --rMax 3 --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin 0 --rMax 3 --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90_observed.root --output plots/scan_total_combined --others "/t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed" --pdf "n" --outRoot "n"
rm higgsCombineTest.MultiDimFit.mH90_observed.root
rm higgsCombineTest.MultiDimFit.mH90_expected.root
#plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90.root --output scan_total
# Expected
#plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90.root --output scan_total_expected --main-label "Expected"


#cd /t3home/gcelotto/ggHbb/abcd/combineTry/
text2workspace.py $datacard -m 90
combineTool.py -M Impacts -d datacards/shapeZdatacard_total_reduced.root  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacards/shapeZdatacard_total_reduced.root  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d datacards/shapeZdatacard_total_reduced.root  -m 90 -o impacts_total.json
plotImpacts.py -i impacts_total.json -o impacts_total
mv impacts_total.p* plots/
rm /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.root
# Run a fit with all nuisance parameters floating and store the workspace in an output file
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.root -M MultiDimFit --saveWorkspace -n zbb.postfit -m 90 --points 100
## Run a scan from the postfit workspace
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit -n zbb.total --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4 -m 90 --points 100
## Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeNuisanceGroups theory -n zbb.freeze_theory -m 90 --points 100
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeParameters allConstrainedNuisances -n zbb.freeze_all -m 90
#
#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.total.MultiDimFit.mH90.root --main-label "Total Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.freeze_all.MultiDimFit.mH90.root:"Stat only":2  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat" 
#mv breakdown.p* plots/
#
