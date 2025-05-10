datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_Z_cat2p0.txt"
rmin=0.3
rmax=1.7
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"
cd $outputCombine


combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
cd $plotFolder
plot1DScan.py --POI r $outputCombine"/higgsCombineTest.MultiDimFit.mH90_observed.root" --output ZObs_cat2p0 --others $outputCombine"/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed" --pdf "n" --outRoot "n"



cd $outputCombine
text2workspace.py $datacard -m 90
workspace="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_Z_cat2p0.root"
combineTool.py -M Impacts -d $workspace  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 -o impacts_total.json
cd $plotFolder
plotImpacts.py -i $outputCombine/impacts_total.json -o impacts_total_cat2p0
mv impacts_total.p* $plotFolder