python /t3home/gcelotto/ggHbb/newFit/afterNN/zPeakSystematics.py -i 0
python /t3home/gcelotto/ggHbb/newFit/afterNN/bkgPlusZ_fit.py -i 0 -w 1 -l 0
python /t3home/gcelotto/ggHbb/newFit/afterNN/plotSystematicVariation.py -i 0 -b Higgs
bash /t3home/gcelotto/ggHbb/newFit/afterNN/scripts/higgsFit/likelihoodScan_cat1.sh

