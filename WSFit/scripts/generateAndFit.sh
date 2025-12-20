signalExpected=$1
CATEGORY=$2
nToys=$3
npdf=$4
ext=Signal$signalExpected
rMin=-10
rMax=12
# Summary arguments
echo "Signal expected is $signalExpected"
echo "Category is $CATEGORY"
echo "Number of toys is $nToys"
cd /t3home/gcelotto/ggHbb/WSFit/datacards
# Generate toys with 3 different PDFS
for toyGenIdx in $(seq 0 $((npdf - 1))); do
        combine -M GenerateOnly -d datacardMulti$CATEGORY.txt  --setParameterRanges r=$rMin,$rMax   --freezeParameters pdfindex_$CATEGORY"_2016_13TeV" --setParameters pdfindex_$CATEGORY"_2016_13TeV"=$toyGenIdx -n ToysPdf$toyGenIdx"_"$ext"_cat"$CATEGORY -m 125   -t $nToys --expectSignal $signalExpected --saveToys
done




# Fit the toys with all 3 different PDFS
for toyGenIdx in $(seq 0 $((npdf - 1))); do
        for pdfFitIdx in $(seq 0 $((npdf - 1))); do
                echo "Fitting toys generated with PDF index "$toyGenIdx" using PDF index "$pdfFitIdx
                combine -M FitDiagnostics -d datacardMulti$CATEGORY.txt  \
                        --toysFile "higgsCombineToysPdf"$toyGenIdx"_"$ext"_cat"$CATEGORY".GenerateOnly.mH125.123456.root" \
                        --setParameterRanges r=$rMin,$rMax  -m 125  \
                        --setParameters pdfindex_$CATEGORY"_"2016_13TeV=$pdfFitIdx  \
                        --freezeParameters pdfindex_$CATEGORY"_"2016_13TeV\
                        --cminDefaultMinimizerStrategy=0 \
                        --X-rtd MINIMIZER_freezeDisassociatedParams  --rMin $rMin --rMax $rMax \
                        -t $nToys \
                        -n "fitPdf"$pdfFitIdx"_ToysGen"$toyGenIdx"_"$ext"_cat"$CATEGORY 

        done
done
/work/gcelotto/miniconda3/envs/myenv/bin/python3 /t3home/gcelotto/ggHbb/WSFit/scripts/generateAndFit.py --cat $CATEGORY --signalExpected $signalExpected --npdf $npdf