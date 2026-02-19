#!/usr/bin/env bash
# Default values

INCLUDE=0
INCLUDEZ=1

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--includeTurnOn)
      INCLUDE="$2"
      shift 2
      ;;
    -z|--z)
      INCLUDEZ="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

#conda activate myenv

/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c 0 # produces step1/ws$CAT_syst.root
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c 1 # produces step1/ws$CAT_syst.root
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c 2 # produces step1/ws$CAT_syst.root
#exit 0
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step2_ws.py -c 0
python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step2_ws.py -c 1
python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step2_ws.py -c 2
#
#
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background
rm /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/fTest_5families_functionalities_cat1/*.png
rm /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/fTest_5families_functionalities_cat2/*.png

./bin/fTest     --infilename "/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws0_nominal.root"  \
                --ncats 1 --singleCat 1 --catNumber 0 --includeTurnOn $INCLUDE --includeZ $INCLUDEZ   \
                --outDir plots/fTest_5families_functionalities_cat0 \
                --iterativeFit 0 --blindSignalRegion 1 \
                --saveMultiPdf /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_0.root 
./bin/fTest     --infilename "/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws1_nominal.root"  \
                --ncats 1 --singleCat 1 --catNumber 1 --includeTurnOn $INCLUDE --includeZ $INCLUDEZ   \
                --outDir plots/fTest_5families_functionalities_cat1 \
                --iterativeFit 0 --blindSignalRegion 1 \
                --saveMultiPdf /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_1.root 
./bin/fTest     --infilename "/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws2_nominal.root"  \
                --ncats 1 --singleCat 1 --catNumber 2 --includeTurnOn $INCLUDE --includeZ $INCLUDEZ   \
                --outDir plots/fTest_5families_functionalities_cat2 \
                --iterativeFit 0 --blindSignalRegion 1 \
                --saveMultiPdf /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_2.root 

/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step3_addH.py -c 0
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step3_addH.py -c 1
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step3_addH.py -c 2

/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step4_addSyst.py -c 0
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step4_addSyst.py -c 1
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step4_addSyst.py -c 2

cd /t3home/gcelotto/ggHbb/WSFit/datacards
combineCards.py datacardMulti0.txt datacardMulti1.txt datacardMulti2.txt > combined.txt
combine -M FitDiagnostics -d combined.txt -t -1 --expectSignal 1  --X-rtd MINIMIZER_freezeDisassociatedParams --setParameterRange r=-30,30