
#!/usr/bin/env bash

# Default values
CATEGORY=2
INCLUDE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--category)
      CATEGORY="$2"
      shift 2
      ;;
    -i|--includeTurnOn)
      INCLUDE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/newFit/afterNN/zPeakSystematics.py -c $CATEGORY -p Z
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/newFit/afterNN/zPeakSystematics.py -c $CATEGORY -p H
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/step1_ws.py -c $CATEGORY
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
python3 /t3home/gcelotto/ggHbb/WSFit/step2_ws.py -c $CATEGORY
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background
rm /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/fTest_5families_functionalities_cat$CATEGORY/*.png
./bin/fTest     --infilename "/t3home/gcelotto/ggHbb/WSFit/ws/step2/ws"$CATEGORY".root"  \
                --ncats 1 --singleCat 1 --catNumber $CATEGORY --includeTurnOn $INCLUDE --includeZ 1 --rebinFactor 1  \
                --outDir plots/fTest_5families_functionalities_cat$CATEGORY \
                --iterativeFit 0 --blindSignalRegion 1 \
                --saveMultiPdf /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_$CATEGORY.root 

python3 /t3home/gcelotto/ggHbb/WSFit/step3_addH.py $CATEGORY
