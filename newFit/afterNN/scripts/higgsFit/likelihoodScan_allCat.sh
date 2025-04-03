dc_cat0="/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/datacards/datacard_H.txt"
dc_cat2p0="/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/datacards/datacard_H.txt"
dc_cat2p1="/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/datacards/datacard_H.txt"
datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/higgsFit/combDC.txt"
combineCards.py $dc_cat0 $dc_cat2p0 $dc_cat2p1 > $datacard

rmin=-3
rmax=5
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"
cd $outputCombine
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125  -n hbb_total.expected
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125 -n hbb_total_statOnly.expected --freezeParameters all --saveWorkspace
cd $plotFolder
plot1DScan.py $outputCombine"/higgsCombinehbb_total.expected.MultiDimFit.mH125.root" --others $outputCombine"/higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root:Expected Stat Only:2" --output HExp_allCat --main-label "Expected"



