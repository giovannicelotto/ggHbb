#!/usr/bin/env bash
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards
combineCards.py datacardMulti0.txt datacardMulti1.txt datacardMulti2.txt > combined.txt
combine -M FitDiagnostics -d combined.txt -t -1 --expectSignal 1  --X-rtd MINIMIZER_freezeDisassociatedParams --setParameterRange r=-30,30