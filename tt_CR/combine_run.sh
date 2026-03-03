cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/tt_CR
combine -M FitDiagnostics datacard_ttbar_CR.txt --redefineSignalPOIs SF_NN  --plots --freezeParameters r


text2workspace.py datacard_ttbar_CR.txt  -o datacard_ttbar_CR.root 
combineTool.py -M Impacts -d datacard_ttbar_CR.root -m 0  --freezeParameters r -n .impacts  --redefineSignalPOIs SF_NN --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN --doFits --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN -o impacts.json
plotImpacts.py -i impacts.json -o impacts