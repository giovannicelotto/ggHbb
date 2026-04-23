# Fit ttbar Control region to derive the scale factor
CATEGORY=$1
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/tt_CR/analysis/datacards

text2workspace.py  datacard_ttbar_CR_$CATEGORY.txt
echo "**************************\n       Performing Fit in Control Region       \n**************************\n"
combine -M FitDiagnostics datacard_ttbar_CR_$CATEGORY.root --redefineSignalPOIs SF_NN  --plots --freezeParameters r -n cat$CATEGORY --robustFit 1 >  /t3home/gcelotto/ggHbb/tt_CR/plots/fitDiagnostics/FitDiagnostics_ttbar_CR_cat$CATEGORY.txt 
OUTDIR="/t3home/gcelotto/ggHbb/tt_CR/plots/prefit/"$CATEGORY
mkdir -p "$OUTDIR"
mv bin1_CMS_th1x_prefit.png $OUTDIR"/prefit/bin1_CMS_th1x_prefit_cat"${CATEGORY}".png"
mv bin1_CMS_th1x_fit_s.png $OUTDIR"/postfit/bin1_CMS_th1x_postfit_cat"${CATEGORY}".png"

#if the plots are not produced:
#combine -M FitDiagnostics datacard_ttbar_CR_$CATEGORY.txt --redefineSignalPOIs SF_NN  --freezeParameters r -n cat$CATEGORY --robustFit 1 >  /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/FitDiagnostics/FitDiagnostics_ttbar_CR_cat$CATEGORY.txt




#Derive impacts ont he ttbar control region

echo "**************************\n       Impacts  for Control region     \n**************************\n"

text2workspace.py datacard_ttbar_CR_$CATEGORY.txt  -o datacard_ttbar_CR_$CATEGORY.root 
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --setParameters r=1 --setParameterRanges r=0.9999,1.00001 --freezeParameters r -n .impacts  --redefineSignalPOIs SF_NN --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --setParameters r=1 --setParameterRanges r=0.9999,1.00001 --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN --doFits --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN -o /t3home/gcelotto/ggHbb/tt_CR/plots/impacts/impacts_$CATEGORY.json
cd /t3home/gcelotto/ggHbb/tt_CR/plots/impacts
plotImpacts.py -i impacts_$CATEGORY.json -o impacts_$CATEGORY
cd /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm
rm /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/higgsCombine_*Fit_.impacts.*.root
rm /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/roostats*.root
