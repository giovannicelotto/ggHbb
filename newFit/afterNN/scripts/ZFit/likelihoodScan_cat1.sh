rmin=0
rmax=2

datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/datacards/datacard_Z.txt"
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"

cd $outputCombine

combine -M FitDiagnostics $datacard --saveNormalizations --saveShapes --redefineSignalPOIs r --robustFit 1 --mass 90
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_observed.root
combine -M MultiDimFit $datacard --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 90 --expectSignal 1 -t -1 
mv higgsCombineTest.MultiDimFit.mH90.root higgsCombineTest.MultiDimFit.mH90_expected.root
cd $plotFolder
plot1DScan.py --POI r $outputCombine"/higgsCombineTest.MultiDimFit.mH90_observed.root" --output ZObs_cat1 --others $outputCombine"/higgsCombineTest.MultiDimFit.mH90_expected.root:Expected:2" --main-label "Observed" --pdf "n" --outRoot "n"




cd $outputCombine
text2workspace.py $datacard -m 90
workspace="/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/datacards/datacard_Z.root"
combineTool.py -M Impacts -d $workspace  -m 90 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 --doFits --robustFit 1
combineTool.py -M Impacts -d $workspace  -m 90 -o impacts_total.json
cd $plotFolder
plotImpacts.py -i $outputCombine/impacts_total.json -o impacts_total_cat1
mv impacts_total.p* $plotFolder
#rm /t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_total.root