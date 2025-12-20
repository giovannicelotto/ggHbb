rm *.root
#text2workspace.py datacardMulti$CATEGORY.txt
#combine -m 123 -M MultiDimFit --saveWorkspace -n teststep1 datacardMulti0.root  --verbose 0 --freezeParameters lumi,rateZbb --X-rtd MINIMIZER_freezeDisassociatedParams  --setParameterRange -5,5
# Default Scan
CATEGORY=$1
DATACARD=datacardMulti$CATEGORY.txt
#datacardMulti$CATEGORY.txt
combine -m 123 -M MultiDimFit \
    -d  $DATACARD\
    -n teststep2_default \
    --expectSignal 1 \
    -t -1 \
    --algo grid \
    --points 30 \
    --rMin -8 --rMax 15 \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --verbose 0\
    --saveNLL \
# RateZbb
combine -m 123 -M MultiDimFit \
    -d $DATACARD \
    -n teststep2_rateZbbFrozen \
    --expectSignal 1 \
    -t -1 \
    --algo grid \
    --points 30 \
    --rMin -8 --rMax 15 \
    --freezeParameters rateZbb \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --verbose 0\
    --saveNLL
#PDFIndex
#combine -m 123 -M MultiDimFit \
#    -d DATACARD \
#    -n teststep2_pdfindexFrozen \
#    --expectSignal 1 \
#    -t -1 \
#    --algo grid \
#    --points 30 \
#    --rMin -4 --rMax 6 \
#    --freezeParameters pdfindex_10_2016_13TeV \
#    --X-rtd MINIMIZER_freezeDisassociatedParams \
#    --verbose 3\
#    --saveNLL
#
plot1DScan.py \
    higgsCombineteststep2_default.MultiDimFit.mH123.root \
    --POI r \
    --main-label "default scan" \
    --others higgsCombineteststep2_rateZbbFrozen.MultiDimFit.mH123.root:"rateZbb frozen":2 \
    --main-color 1 \
    --output r_scan_cat$CATEGORY.png
    #--others higgsCombineteststep2_rateZbbFrozen.MultiDimFit.mH123.root:"rateZbb frozen":2 \ #higgsCombineteststep2_pdfindexFrozen.MultiDimFit.mH123.root:"pdfindex fixed":3 \