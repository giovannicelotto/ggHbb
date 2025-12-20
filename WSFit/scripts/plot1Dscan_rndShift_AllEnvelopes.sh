
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti0.txt datacardMulti10.txt datacardMulti100.txt > combined_0_10_100.txt
combine -M MultiDimFit -d combined_0_10_100.txt \
        -m 125 --setParameters pdfindex_0_2016_13TeV=-1:pdfindex_10_2016_13TeV=-1:pdfindex_100_2016_13TeV=-1 \
        --algo grid --rMin -10 --rMax 12 --points 100  \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
         -n fitData_Blind_CatCombined_pdfDiscrete

python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_AllEnvelopes.py