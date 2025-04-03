datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/datacards/datacard_H.txt"
rmin=-3
rmax=5
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
cd $outputCombine

combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125  -n hbb_total.expected
combine -M MultiDimFit $datacard --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax  --mass 125 -n hbb_total_statOnly.expected --freezeParameters all --saveWorkspace


cd /t3home/gcelotto/ggHbb/newFit/afterNN/
plot1DScan.py $outputCombine/higgsCombinehbb_total.expected.MultiDimFit.mH125.root   --others $outputCombine"/higgsCombinehbb_total_statOnly.expected.MultiDimFit.mH125.root:Expected Stat Only:2" \
    --output plots/HiggsExp_cat2p0.png --main-label "Expected"