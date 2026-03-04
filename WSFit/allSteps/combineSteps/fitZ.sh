
#!/usr/bin/env bash
# Default values
CATEGORY=2
INCLUDE=0
INCLUDEZ=1

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

conda activate myenv
rm /t3home/gcelotto/ggHbb/WSFit/ws/step1/ws"$CATEGORY"*.root
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY # produces step1/ws$CAT_syst.root
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst puid_up
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst puid_down
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst scale
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst PS_ISR
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst PS_FSR
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst alphaS
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst btag_hf_up
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst btag_hf_down
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst btag_lightf_up
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/step1_ws.py -c $CATEGORY --syst btag_lightf_down

#
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
rm /t3home/gcelotto/ggHbb/WSFit/ws/step2/ws$CATEGORY".root"
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step2_ws.py -c $CATEGORY
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background
rm /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/fTest_5families_functionalities_cat$CATEGORY/*.png
rm /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_$CATEGORY.root
./bin/fTest     --infilename "/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws"$CATEGORY"_nominal.root"  \
                --ncats 1 --singleCat 1 --catNumber $CATEGORY --includeTurnOn $INCLUDE --includeZ $INCLUDEZ   \
                --outDir plots/fTest_5families_functionalities_cat$CATEGORY \
                --iterativeFit 0 --blindSignalRegion 1 \
                --saveMultiPdf /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_$CATEGORY.root 

echo "Step 2 done, now adding H and systematics"
rm /t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/multipdf_$CATEGORY".root"
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step3_addH.py -c $CATEGORY
/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre4/external/el9_amd64_gcc12/bin/python3 /t3home/gcelotto/ggHbb/WSFit/allSteps/step4_addSyst.py -c $CATEGORY
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /t3home/gcelotto/.bashrc
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/tt_CR/prepareDataset_forFitSF.py --category $CATEGORY



# Fit ttbar Control region to derive the scale factor
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
cd /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm
combine -M FitDiagnostics datacard_ttbar_CR_$CATEGORY.txt --redefineSignalPOIs SF_NN  --plots --freezeParameters r -n cat$CATEGORY --robustFit 1


text2workspace.py datacard_ttbar_CR_$CATEGORY.txt  -o datacard_ttbar_CR_$CATEGORY.root 
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts  --redefineSignalPOIs SF_NN --doInitialFit --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN --doFits --robustFit 1
combineTool.py -M Impacts -d datacard_ttbar_CR_$CATEGORY.root -m 0  --freezeParameters r -n .impacts --redefineSignalPOIs SF_NN -o impacts_$CATEGORY.json
plotImpacts.py -i impacts_$CATEGORY.json -o impacts_$CATEGORY



# Generate Datacard
/work/gcelotto/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/WSFit/allSteps/generateDatacard.py --output "/t3home/gcelotto/ggHbb/WSFit/datacards/datacardMulti"$CATEGORY".txt" --ws_file "/t3home/gcelotto/ggHbb/WSFit/ws/step5/ws"$CATEGORY".root" --cat $CATEGORY --ws_name "ws3" --processes Z --lumi 1.025



## Plot Likelihood Scan per Category
bash /t3home/gcelotto/ggHbb/WSFit/scripts/plot1Dscan_rndShift_PerCat.sh  $CATEGORY

#PRint Signal Strength With Uncertainty from Best PDF on Asimov
cd /t3home/gcelotto/ggHbb/WSFit/datacards
combine -M FitDiagnostics -d "datacardMulti"$CATEGORY".txt" -t -1 --expectSignal 1  --X-rtd MINIMIZER_freezeDisassociatedParams --setParameterRange r=-30,30:rateZbb=-3,5 --freezeParameters "pdfindex_"$CATEGORY"_2016_13TeV" --cminDefaultMinimizerStrategy 0   > fitDiagnostics_expected/fitDiagnostics_expected_cat$CATEGORY.txt
#
#
## Print impacts
text2workspace.py "datacardMulti"$CATEGORY".txt" -m 125
combineTool.py -M Impacts -d "datacardMulti"$CATEGORY".root" -m 125 -n .impacts --setParameterRanges r=-30,30 --doInitialFit --robustFit 1 --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 \
                        --freezeParameters "pdfindex_"$CATEGORY"_2016_13TeV" --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d "datacardMulti"$CATEGORY".root" -m 125 -n .impacts --setParameterRanges r=-30,30 --doFits --robustFit 1  --X-rtd MINIMIZER_freezeDisassociatedParams -t -1 --expectSignal 1 \
                        --freezeParameters "pdfindex_"$CATEGORY"_2016_13TeV" --cminDefaultMinimizerStrategy 0 
combineTool.py -M Impacts -d "datacardMulti"$CATEGORY".root" -m 125 -n .impacts --setParameterRanges r=-30,30 -o "impacts/impacts_datacardMulti"$CATEGORY".json"
plotImpacts.py -i "impacts/impacts_datacardMulti"$CATEGORY".json" -o "impacts/impacts_datacardMulti"$CATEGORY
rm /t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombine_paramFit_.impacts_*.root
rm /t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinefitData_Blind_Cat*.root
rm /t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinerateZbbScan_*.root