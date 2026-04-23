# Arguments     : signalExpected CATEGORY nToys npdf
signalExpected=$1
CATEGORY=$2
nToys=$3
npdf=$4
ext=Signal$signalExpected
rMin=-30
rMax=32
# Summary arguments
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv

echo "Signal expected is $signalExpected"
echo "Category is $CATEGORY"
echo "Number of toys is $nToys"
cd /t3home/gcelotto/ggHbb/WSFit/datacards
# Generate toys with 3 different PDFS
for toyGenIdx in $(seq 0 $((npdf - 1))); do
        combine -M GenerateOnly -d datacardMulti$CATEGORY.txt  --setParameterRanges r=$rMin,$rMax   --freezeParameters pdfindex_$CATEGORY"_2016_13TeV" --setParameters pdfindex_$CATEGORY"_2016_13TeV"=$toyGenIdx -n ToysPdf$toyGenIdx"_"$ext"_cat"$CATEGORY -m 125   -t $nToys --expectSignal $signalExpected --saveToys
done
#
#
#
## Fit the toys with all 3 different PDFS
for toyGenIdx in $(seq 0 $((npdf - 1))); do

        echo "Fitting toys generated with PDF index "$toyGenIdx" using MultiPDF "
        combine -M FitDiagnostics -d datacardMulti$CATEGORY.txt  \
                --toysFile "higgsCombineToysPdf"$toyGenIdx"_"$ext"_cat"$CATEGORY".GenerateOnly.mH125.123456.root" \
                --setParameterRanges r=$rMin,$rMax -m 125 \
                --setParameters pdfindex_${CATEGORY}_2016_13TeV=-1  \
                --X-rtd MINIMIZER_freezeDisassociatedParams  --rMin $rMin --rMax 9 \
                -t $nToys \
                -n "fitPdfEnvelope_ToysGen"$toyGenIdx"_"$ext"_cat"$CATEGORY 
done

/work/gcelotto/miniconda3/envs/myenv/bin/python3 /t3home/gcelotto/ggHbb/WSFit/scripts/generateAndFitEnvelope.py --cat $CATEGORY --signalExpected $signalExpected --npdf $npdf