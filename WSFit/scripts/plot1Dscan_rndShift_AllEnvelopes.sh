cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti4.txt \
                datacardMulti2.txt \
                datacardMulti10.txt \
                datacardMulti11.txt \
                datacardMulti12.txt  > combined.txt
combine -M MultiDimFit -d combined.txt \
       -m 125 \
       --algo grid --rMin -6 --rMax 8 --points 30  \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 \
         -n fitData_Blind_CatCombined_pdfDiscrete  
python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_AllEnvelopes.py