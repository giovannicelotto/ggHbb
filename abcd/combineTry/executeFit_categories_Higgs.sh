cd datacards/
combineCards.py ch1=shapeHdatacard_lcPass.txt ch2=shapeHdatacard_lcFail_b2Med.txt ch3=shapeHdatacard_lcFail_b2Tight.txt > combined_datacard_higgs.txt
cd ..
datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/combined_datacard_higgs.txt"
combine -M AsymptoticLimits $datacard --rMin -10 --rMax 10 --mass 125 --run expected -n hbb_categories.expected

combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin -10 --rMax 10 --mass 125  -n hbb_categories.expected
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin -10 --rMax 10 --mass 125 -n hbb_categories_statOnly.expected --freezeParameters all --saveWorkspace
#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb_categories_statOnly.expected.MultiDimFit.mH125.root -o scan_statOnly_H125_expected --main-label "Stat Only"
plot1DScan.py higgsCombinehbb_categories.expected.MultiDimFit.mH125.root   --others "higgsCombinehbb_categories_statOnly.expected.MultiDimFit.mH125.root:Expected Stat Only:2" \
    -o scan_categories_H125_expected --main-label "Expected"


#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinehbb_categories.expected.MultiDimFit.mH125.root -o scan_categories_H125_expected --main-label "Expected"


#text2workspace.py $datacard -m 125
#combineTool.py -M Impacts -d datacards/shapeZdatacard_categories.root  -m 125 --doInitialFit --robustFit 1
#combineTool.py -M Impacts -d datacards/shapeZdatacard_categories.root  -m 125 --doFits --robustFit 1
#combineTool.py -M Impacts -d datacards/shapeZdatacard_categories.root  -m 125 -o impacts_categories.json
#plotImpacts.py -i impacts_categories.json -o impacts_categories

# Run a fit with all nuisance parameters floating and store the workspace in an output file
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_categories.root -M MultiDimFit --saveWorkspace -n zbb.postfit -m 125 --points 100
# Run a scan from the postfit workspace
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH125.root -M MultiDimFit -n zbb.categories --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4 -m 125 --points 100
# Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH125.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeNuisanceGroups theory -n zbb.freeze_theory -m 125 --points 100
#combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH125.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeParameters allConstrainedNuisances -n zbb.freeze_all -m 125
#plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.categories.MultiDimFit.mH125.root --main-label "categories Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.freeze_all.MultiDimFit.mH125.root:"Stat only":2  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat" 

