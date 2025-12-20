#!/bin/bash
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
freezeParamsBern5=$(
  cat <<EOF
env_pdf_Exponential_3_cat10_exp3_p1
env_pdf_Exponential_3_cat10_exp3_p2
env_pdf_Exponential_3_cat10_exp3_f1
env_pdf_Exponential_3_cat10_z_norm
shapeSig_signal_Cat1__norm
pdfindex_10_2016_13TeV
EOF
)

freezeParamsExpo3=$(
  cat <<EOF
env_pdf_Bernstein_5_cat10_bern5_p0
env_pdf_Bernstein_5_cat10_bern5_p1
env_pdf_Bernstein_5_cat10_bern5_p2
env_pdf_Bernstein_5_cat10_bern5_p3
env_pdf_Bernstein_5_cat10_bern5_p4
env_pdf_Bernstein_5_cat10_z_norm
shapeSig_signal_Cat1__norm
pdfindex_10_2016_13TeV
EOF
)





freezeParamsBern5=$(echo "$freezeParamsBern5" | paste -sd, -)
freezeParamsExpo3=$(echo "$freezeParamsExpo3" | paste -sd, -)

cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20
# ****************************************
#
# PDF 0
# Generate Toys based on pdf 0
# # ****************************************


combine -M MultiDimFit \
  -d datacardMulti10.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_10_2016_13TeV=0 \
  -n fixed_pdfCat10_0 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsBern5" 
# ****************************************
# # PDF 1
# # ****************************************

  # Fit based on pdf 1
combine -M MultiDimFit \
  -d datacardMulti10.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-10,10 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_10_2016_13TeV=1 \
  -n fixed_pdfCat10_1 \
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsExpo3"
  
