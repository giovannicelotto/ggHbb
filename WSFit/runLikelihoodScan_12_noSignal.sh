#!/bin/bash
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
freezeParamsBern2=$(
  cat <<EOF
env_pdf_Bernstein_3_cat12_bern3_p0
env_pdf_Bernstein_3_cat12_bern3_p1
env_pdf_Bernstein_3_cat12_bern3_p2
env_pdf_Bernstein_3_cat12_z_norm
env_pdf_Exponential_1_cat12_exp1_p1
env_pdf_Exponential_1_cat12_z_norm
shapeSig_signal_Cat12__norm
pdfindex_12_2016_13TeV
EOF
)

freezeParamsBern3=$(
  cat <<EOF
env_pdf_Bernstein_2_cat12_bern2_p0
env_pdf_Bernstein_2_cat12_bern2_p1
env_pdf_Bernstein_2_cat12_z_norm
env_pdf_Exponential_1_cat12_exp1_p1
env_pdf_Exponential_1_cat12_z_norm
shapeSig_signal_Cat12__norm
pdfindex_12_2016_13TeV
EOF
)

freezeParamsExpo1=$(
  cat <<EOF
env_pdf_Bernstein_2_cat12_bern2_p0
env_pdf_Bernstein_2_cat12_bern2_p1
env_pdf_Bernstein_2_cat12_z_norm
env_pdf_Bernstein_3_cat12_bern3_p0
env_pdf_Bernstein_3_cat12_bern3_p1
env_pdf_Bernstein_3_cat12_bern3_p2
env_pdf_Bernstein_3_cat12_z_norm
shapeSig_signal_Cat12__norm
pdfindex_12_2016_13TeV
EOF
)



freezeParamsExpo1=$(echo "$freezeParamsExpo1" | paste -sd, -)
freezeParamsBern3=$(echo "$freezeParamsBern3" | paste -sd, -)
freezeParamsBern2=$(echo "$freezeParamsBern2" | paste -sd, -)

cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20
# ****************************************
#
# PDF 0
# Generate Toys based on pdf 0
# # ****************************************


combine -M MultiDimFit \
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_12_2016_13TeV=0 \
  -n fixed_pdfCat12_0 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsBern2" 
# ****************************************
# # PDF 1
# # ****************************************

  # Fit based on pdf 1
combine -M MultiDimFit \
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_12_2016_13TeV=1 \
  -n fixed_pdfCat12_1 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsBern3" 
  
combine -M MultiDimFit \
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_12_2016_13TeV=2 \
  -n fixed_pdfCat12_2 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsExpo1" 