cd datacards/
combineCards.py ch1=shapeZdatacard_ch1.txt ch2=shapeZdatacard_ch2.txt ch3=shapeZdatacard_ch3.txt ch4=shapeZdatacard_ch4.txt ch5=shapeZdatacard_ch5.txt ch6=shapeZdatacard_ch6.txt > combined_datacard.txt
cd ..
combine -M MultiDimFit datacards/combined_datacard.txt --algo grid --points 100 --rMin -0 --rMax 2
python3 /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/scripts/plot1DScan.py --POI r /t3home/gcelotto/ggHbb/abcd/combineTry/higgsCombineTest.MultiDimFit.mH120.root