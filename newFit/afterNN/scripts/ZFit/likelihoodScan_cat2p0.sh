cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_Z_cat2p0.txt"
ws="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_Z_cat2p0.root"
rmin=-3
rmax=5
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"
cd $outputCombine

#Create ws
text2workspace.py $datacard  -o $ws

combine -M MultiDimFit $ws --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
combine -M MultiDimFit $ws --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
cd $plotFolder
plot1DScan.py --POI r $outputCombine"/higgsCombineTest.MultiDimFit.mH90_observed.root" --output ZObs_cat2p0 --others $outputCombine"/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed" --pdf "n" --outRoot "n"


exit


cd $outputCombine
text2workspace.py $datacard -m 90
workspace="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_Z_cat2p0.root"
combineTool.py -M Impacts -d $workspace  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 -o impacts_total.json
cd $plotFolder
plotImpacts.py -i $outputCombine/impacts_total.json -o impacts_total_cat2p0
mv impacts_total.p* $plotFolder