category=$1
rMin=-40
rMax=40
points=30
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        --setParameterRanges pdfindex_$category"_2016_13TeV"=0,6:rateZbb=-1,3 \
        -m 125 --setParameters pdfindex_$category"_2016_13TeV"=0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdfDiscrete"

combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=0 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf0"
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=1 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf1"
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=2 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf2"
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=3 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf3"
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=4 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf4"
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=5 \
        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
        --setParameterRanges rateZbb=-1,3\
        --cminDefaultMinimizerStrategy 0 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdf5"
#if [ "$category" = "10" ]; then
#        combine -M MultiDimFit -d "datacardMulti"$category".txt" \
#        -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=3 \
#        --freezeParameters "pdfindex_"$category"_2016_13TeV" \
#        --cminDefaultMinimizerStrategy 0 \
#        --algo grid --rMin $rMin --rMax $rMax --points 100 \
#        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
#        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
#        -n fitData_Blind_Cat$category"pdf3"
#fi

#Then run
/work/gcelotto/miniconda3/envs/myenv/bin/python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_PerCat.py --idx $category     