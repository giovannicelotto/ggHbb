datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeHdatacard_total.txt"
combine -M AsymptoticLimits $datacard --rMin -10 --rMax 10 --mass 125 --run expected -n hbb_total.expected

combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin -10 --rMax 10 --mass 125  -n hbb_total.expected
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin -10 --rMax 10 --mass 125 -n hbb_total_statOnly.expected --freezeParameters all --saveWorkspace
#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root -o scan_statOnly_H125_expected --main-label "Stat Only"
plot1DScan.py higgsCombinehbb_total.expected.MultiDimFit.mH125.root   --others "higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root:Expected Stat Only:2" \
    -o scan_comparison_H125_expected --main-label "Expected"
mv scan_comparison*.p* plots/

#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb_total.expected.MultiDimFit.mH125.root -o scan_total_H125_expected --main-label "Expected"


#text2workspace.py $datacard -m 125
#combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 125 --doInitialFit --robustFit 1
#combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 125 --doFits --robustFit 1
#combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 125 -o impacts_inclusive_Higgs.json
#plotImpacts.py -i impacts_inclusive_Higgs.json -o impacts_inclusive_higgs
#
## Run a fit with all nuisance parameters floating and store the workspace in an output file
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeHdatacard_total.root -M MultiDimFit --saveWorkspace -n hbb.postfit -m 125 --points 100 --expectSignal 1 -t -1
## Run a scan from the postfit workspace
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb.postfit.MultiDimFit.mH125.root -M MultiDimFit -n hbb.total --algo grid --snapshotName MultiDimFit --setParameterRanges r=-2,4 -m 125 --points 100 --expectSignal 1 -t -1
## Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb.postfit.MultiDimFit.mH125.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=-2,4  --freezeNuisanceGroups theory -n hbb.freeze_theory -m 125 --points 100  --expectSignal 1 -t -1
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb.postfit.MultiDimFit.mH125.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=-2,4  --freezeParameters allConstrainedNuisances -n hbb.freeze_all -m 125 --expectSignal 1 -t -1
#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb.total.MultiDimFit.mH125.root --main-label "Expected Total Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb.freeze_all.MultiDimFit.mH125.root:"Expected Stat only":2  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat" 
