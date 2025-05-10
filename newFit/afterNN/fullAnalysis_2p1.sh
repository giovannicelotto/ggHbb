python /t3home/gcelotto/ggHbb/newFit/afterNN/zPeakSystematics.py -i 2
python /t3home/gcelotto/ggHbb/newFit/afterNN/bkgPlusZ_fit.py -i 2 -w 1
python /t3home/gcelotto/ggHbb/newFit/afterNN/plotSystematicVariation.py -i 2 -b Higgs
bash /t3home/gcelotto/ggHbb/newFit/afterNN/scripts/higgsFit/likelihoodScan_cat2p1.sh

