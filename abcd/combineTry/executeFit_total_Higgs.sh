cd /t3home/gcelotto/ggHbb/abcd/combineTry
datacard="/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeHdatacard_total.txt"
# Define the dynamic part of the filename (e.g., "2A_1D_3A" or "2A_1D")
dynamic_part="1D"  # Change this value as needed
model="Apr01_1000p0"

# Define the new ROOT filename based on the dynamic part
new_root_file="/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_${model}_dd__${dynamic_part}_corrected.root"

# Use sed to replace the line in place
sed -i "s|shapes \*    total  .*root|shapes *    total  ${new_root_file} \$PROCESS|" "$datacard"



rmin=-2.5
rmax=4.5
combine -M AsymptoticLimits $datacard --rMin $rmin --rMax $rmax --mass 125 --run expected -n hbb_total.expected

combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125  -n hbb_total.expected
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125 -n hbb_total_statOnly.expected --freezeParameters all --saveWorkspace
#plot1DScan.py higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root --output plots/scan_H125_expected_stat --main-label "Stat Only"
plot1DScan.py higgsCombinehbb_total.expected.MultiDimFit.mH125.root   --others "higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root:Expected Stat Only:2" \
    --output plots/scan_H125_expected --main-label "Expected"

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

