category=$1
echo "Running for category " $category
rMin=-40
rMax=40
points=40
n=$(python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_PerCat.py --idx $category --onlyNumber 1)
echo $n " pdfs found for category " $category
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards
combine -M MultiDimFit -d "datacardMulti"$category".txt" \
        --setParameterRanges pdfindex_$category"_2016_13TeV"=0,$n:rateZbb=-1,3 \
        -m 125 --setParameters pdfindex_$category"_2016_13TeV"=2 \
        --algo grid --rMin $rMin --rMax $rMax --points $points \
        --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
        --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
        -n fitData_Blind_Cat$category"pdfDiscrete"
#
#

for ((i=0; i<$n; i++)); do
        combine -M MultiDimFit -d "datacardMulti"$category".txt" \
                -m 125 --setParameters "pdfindex_"$category"_2016_13TeV"=$i \
                --freezeParameters "pdfindex_"$category"_2016_13TeV" \
                --setParameterRanges rateZbb=-1,3\
                --cminDefaultMinimizerStrategy 0 \
                --algo grid --rMin $rMin --rMax $rMax --points $points \
                --saveNLL --X-rtd MINIMIZER_freezeDisassociatedParams \
                --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
                -n fitData_Blind_Cat$category"pdf"$i
done

#Then run
/work/gcelotto/miniconda3/envs/myenv/bin/python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_PerCat.py --idx $category     