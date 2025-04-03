datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/datacards/datacard_Z.txt"
rmin=0
rmax=4
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"
cd $outputCombine


combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
cd $plotFolder
plot1DScan.py --POI r $outputCombine"/higgsCombineTest.MultiDimFit.mH90_observed.root" --output ZObs_cat2p1 --others $outputCombine"/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed" --pdf "n" --outRoot "n"




cd $outputCombine
text2workspace.py $datacard -m 90
workspace="/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/datacards/datacard_Z.root"
# check if jet1_btag has well defined constraint
echo "Mycheck"
combine -M MultiDimFit -n _checkConstraints --algo none -m 90 -d $workspace
echo "Mycheck"
combineTool.py -M Impacts -d $workspace -m 90 --printParameters
echo "Double chekc"
combineTool.py -M Impacts -d $workspace  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 -o impacts_total.json
cd $plotFolder
plotImpacts.py -i $outputCombine/impacts_total.json -o impacts_total_cat2p1
mv impacts_total.p* $plotFolder