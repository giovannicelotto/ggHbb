
#!/usr/bin/env bash
CATEGORY="$1"
echo "Running combine for category $CATEGORY"
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm
combine -M FitDiagnostics datacard_ttbar_CR_$CATEGORY.txt --redefineSignalPOIs SF_NN  --plots --freezeParameters r -n cat$CATEGORY --robustFit 1


text2workspace.py datacard_ttbar_CR_$CATEGORY.txt  -o datacard_ttbar_CR_$CATEGORY.root 
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts  --redefineSignalPOIs SF_NN --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN --doFits --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN -o impacts_$CATEGORY.json
plotImpacts.py -i impacts_$CATEGORY.json -o impacts_$CATEGORY