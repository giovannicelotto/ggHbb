cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti0.txt \
                datacardMulti1.txt \
                datacardMulti2.txt \
                datacardMulti3.txt \
                datacardMulti4.txt \
                datacardMulti5.txt \
                datacardMulti6.txt \
                datacardMulti10.txt  > combined.txt
combine -M MultiDimFit -d combined.txt \
        -m 125 \
        --algo grid --rMin -6 --rMax 8 --points 30  \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
         -n fitData_Blind_CatCombined_pdfDiscrete  
python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_AllEnvelopes.py