cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
dc_cat0="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_H_cat1.txt"
dc_cat2p0="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_H_cat2p0.txt"
dc_cat2p1="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/datacard_H_cat2p1.txt"
datacard="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/combDC.txt"
combineCards.py $dc_cat0 $dc_cat2p0 $dc_cat2p1 > $datacard
#Create ws
ws="/t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/combDC.root"
text2workspace.py $datacard  -o $ws


rmin=-3
rmax=5
outputCombine="/t3home/gcelotto/ggHbb/newFit/afterNN/outputCombine"


# move in the folder where temp files will be created
cd $outputCombine
rm *.root
combine -M Significance $ws -t -1 --expectSignal=1



#Expected total
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 200 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_total.expected
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 200 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_statOnly.expected --freezeParameters allConstrainedNuisances 
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 200 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_JEC_Frozen.expected --freezeNuisanceGroups JEC
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 200 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_JEC_JER_Frozen.expected --freezeNuisanceGroups JEC,JER
combine -M MultiDimFit $ws --expectSignal 1 -t -1 --algo grid --points 200 --rMin $rmin --rMax $rmax --mass 125 --robustFit 1 -n hbb_JEC_JER_btag_Frozen.expected --freezeNuisanceGroups JEC,JER,btag
##
#

cd /t3home/gcelotto/ggHbb/newFit/afterNN/
plot1DScan.py $outputCombine/higgsCombinehbb_total.expected.MultiDimFit.mH125.root \
  --main-label "Expected" \
  --others \
    $outputCombine/higgsCombinehbb_statOnly.expected.MultiDimFit.mH125.root:"Stat Only":2 \
  --output plots/HiggsExp_allCat_groups \
  --breakdown "Syst,Stat Only"\
  --pdf "y"

plot1DScan.py $outputCombine/higgsCombinehbb_total.expected.MultiDimFit.mH125.root \
  --main-label "Expected" \
  --others \
    $outputCombine/higgsCombinehbb_JEC_Frozen.expected.MultiDimFit.mH125.root:"JEC Frozen":3\
    $outputCombine/higgsCombinehbb_JEC_JER_Frozen.expected.MultiDimFit.mH125.root:"JEC+JER Frozen":4\
    $outputCombine/higgsCombinehbb_JEC_JER_btag_Frozen.expected.MultiDimFit.mH125.root:"JEC+JER+btag Frozen":5\
    $outputCombine/higgsCombinehbb_statOnly.expected.MultiDimFit.mH125.root:"Stat Only":2 \
  --output plots/HiggsExp_allCat_groups \
  --breakdown "JEC, JER, btag,Rest,Stat Only"\
  --pdf "y"




cd /t3home/gcelotto/ggHbb/newFit/afterNN/scripts/datacards/
cd $outputCombine
plotFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/plots"
combineTool.py -M Impacts -d $ws -m 125 --expectSignal 1 -t -1 --rMin $rmin --rMax $rmax   --robustFit 1 --doInitialFit
combineTool.py -M Impacts -d $ws -m 125 --expectSignal 1 -t -1 --rMin $rmin --rMax $rmax   --robustFit 1 --doFits
combineTool.py -M Impacts -d $ws -m 125 -o impacts_allCat.json
plotImpacts.py -i impacts_allCat.json -o impacts_allCat_Higgs
mv impacts_allCat_Higgs.pdf $plotFolder