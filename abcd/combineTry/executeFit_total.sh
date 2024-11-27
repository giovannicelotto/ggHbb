datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.txt"
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin -0 --rMax 2 --mass 90
python3 /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH90.root -o scan_total


text2workspace.py $datacard -m 90
combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d datacards/shapeZdatacard_total.root  -m 90 -o impacts_total.json
plotImpacts.py -i impacts_total.json -o impacts_total

# Run a fit with all nuisance parameters floating and store the workspace in an output file
combine /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.root -M MultiDimFit --saveWorkspace -n zbb.postfit -m 90 --points 100
# Run a scan from the postfit workspace
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit -n zbb.total --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4 -m 90 --points 100
# Run additional scans using the post-fit workspace, sequentially adding another group to the list of groups to freeze
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeNuisanceGroups theory -n zbb.freeze_theory -m 90 --points 100
combine /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.postfit.MultiDimFit.mH90.root -M MultiDimFit --algo grid --snapshotName MultiDimFit --setParameterRanges r=0,4  --freezeParameters allConstrainedNuisances -n zbb.freeze_all -m 90

plot1DScan.py /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.total.MultiDimFit.mH90.root --main-label "Total Uncert."  --others /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombinezbb.freeze_all.MultiDimFit.mH90.root:"stat only":6  --output breakdown --y-max 10 --y-cut 40 --breakdown "syst,stat" 
#plot1DScan.py higgsCombinezbb.total.MultiDimFit.mH120.root --main-label "Total Uncert."  --others higgsCombinezbb.freeze_all.MultiDimFit.mH120.root:"stat only":6  --output breakdown --y-max 10 --y-cut 40 --breakdown "theory,stat"
#combine -M FitDiagnostics /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.txt
#python3 /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py /t3home/gcelotto/ggHbb/abcd/combineTry/fitDiagnosticsTest.root
