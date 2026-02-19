cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/WSFit/datacards/
combineCards.py datacardMulti0.txt \
                datacardMulti1.txt \
                datacardMulti2.txt \
                datacardMulti10.txt \
                datacardMulti11.txt \
                datacardMulti100.txt \
                datacardMulti101.txt > combined.txt
text2workspace.py combined.txt -m 125
combine -M MultiDimFit combined.root -m 125 -n .scan2D --algo grid --points 800 -t -1 --expectSignal 1\
                        --setParameters pdfindex_0_2016_13TeV=2,pdfindex_1_2016_13TeV=0,pdfindex_2_2016_13TeV=1,pdfindex_10_2016_13TeV=1,pdfindex_11_2016_13TeV=3,pdfindex_100_2016_13TeV=1,pdfindex_101_2016_13TeV=4\
                        --freezeParameters pdfindex_0_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_2_2016_13TeV,pdfindex_10_2016_13TeV,pdfindex_11_2016_13TeV,pdfindex_100_2016_13TeV,pdfindex_101_2016_13TeV --cminDefaultMinimizerStrategy 0 \
                         --X-rtd MINIMIZER_freezeDisassociatedParams -P r -P rateZbb --setParameterRanges r=-6,8:rateZbb=0.3,1.7

python3 /t3home/gcelotto/ggHbb/WSFit/scripts/plot_2D_scan.py