cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_H_cat1.txt"
ws="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_H_cat1.root"
rmin=-3
rmax=5
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"
# move in the folder where temp files will be created
cd $outputCombine


combine -M Significance $datacard -t -1 --expectSignal=1



#Create ws
text2workspace.py $datacard  -o $ws

#Expected total
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1   -n hbb_total.expected
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_statOnly.expected --freezeParameters allConstrainedNuisances

cd /t3home/gcelotto/ggHbb/newFit/afterNN/
plot1DScan.py $outputCombine/higgsCombinehbb_total.expected.MultiDimFit.mH125.root \
  --main-label "Expected" \
  --others \
    $outputCombine/higgsCombinehbb_statOnly.expected.MultiDimFit.mH125.root:"Stat Only":2 \
  --output plots/HiggsExp_cat1_StatOnly \
  --breakdown "Syst,Stat Only"


#combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_JEC_JER_Frozen.expected --freezeNuisanceGroups JEC,JER
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 100 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_JEC_JER_btag_Frozen.expected --freezeNuisanceGroups JEC,JER,btag
#


plot1DScan.py $outputCombine/higgsCombinehbb_total.expected.MultiDimFit.mH125.root \
  --main-label "Expected" \
  --others \
    $outputCombine/higgsCombinehbb_JEC_JER_btag_Frozen.expected.MultiDimFit.mH125.root:"JEC+JER+btag Frozen":8\
    $outputCombine/higgsCombinehbb_statOnly.expected.MultiDimFit.mH125.root:"Stat Only":2 \
  --output plots/HiggsExp_cat1_groups \
  --breakdown "Jets Syst,Bkg Estimation,Stat Only"

    #$outputCombine/higgsCombinehbb_JEC_JER_Frozen.expected.MultiDimFit.mH125.root:"JEC+JER Frozen":6 \


plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"




cd /t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/
text2workspace.py $datacard -m 125
cd $outputCombine
combineTool.py -M Impacts -d $ws -m 125 --expectSignal 1 -t -1 --points 100 --rMin $rmin --rMax $rmax --doInitialFit --robustFit 1
combineTool.py -M Impacts -d $ws -m 125 --expectSignal 1 -t -1 --points 100 --rMin $rmin --rMax $rmax --robustFit 1 --doFits
combineTool.py -M Impacts -d $ws -m 125 -o impacts_2p0.json
plotImpacts.py -i impacts_2p0.json -o impacts_cat1_Higgs
mv impacts_cat1_Higgs.pdf $plotFolder