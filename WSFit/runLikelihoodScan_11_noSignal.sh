#!/bin/bash
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
freezeParamsBern4=$(
  cat <<EOF
env_pdf_Bernstein_5_cat11_bern5_p0
env_pdf_Bernstein_5_cat11_bern5_p1
env_pdf_Bernstein_5_cat11_bern5_p2
env_pdf_Bernstein_5_cat11_bern5_p3
env_pdf_Bernstein_5_cat11_bern5_p4
env_pdf_Bernstein_5_cat11_z_norm
env_pdf_Exponential_3_cat11_exp3_p1
env_pdf_Exponential_3_cat11_exp3_p2
env_pdf_Exponential_3_cat11_exp3_f1
env_pdf_Exponential_3_cat11_z_norm
shapeSig_signal_Cat11__norm
pdfindex_11_2016_13TeV
EOF
)

freezeParamsBern5=$(
  cat <<EOF
env_pdf_Bernstein_4_cat11_bern4_p0
env_pdf_Bernstein_4_cat11_bern4_p1
env_pdf_Bernstein_4_cat11_bern4_p2
env_pdf_Bernstein_4_cat11_bern4_p3
env_pdf_Bernstein_4_cat11_z_norm
env_pdf_Exponential_3_cat11_exp3_p1
env_pdf_Exponential_3_cat11_exp3_p2
env_pdf_Exponential_3_cat11_exp3_f1
env_pdf_Exponential_3_cat11_z_norm
shapeSig_signal_Cat11__norm
pdfindex_11_2016_13TeV
EOF
)

freezeParamsExpo3=$(
  cat <<EOF
env_pdf_Bernstein_5_cat11_bern5_p0
env_pdf_Bernstein_5_cat11_bern5_p1
env_pdf_Bernstein_5_cat11_bern5_p2
env_pdf_Bernstein_5_cat11_bern5_p3
env_pdf_Bernstein_5_cat11_bern5_p4
env_pdf_Bernstein_5_cat11_z_norm
env_pdf_Bernstein_4_cat11_bern4_p0
env_pdf_Bernstein_4_cat11_bern4_p1
env_pdf_Bernstein_4_cat11_bern4_p2
env_pdf_Bernstein_4_cat11_bern4_p3
env_pdf_Bernstein_4_cat11_z_norm
shapeSig_signal_Cat11__norm
pdfindex_11_2016_13TeV
EOF
)



freezeParamsBern4=$(echo "$freezeParamsBern4" | paste -sd, -)
freezeParamsBern5=$(echo "$freezeParamsBern5" | paste -sd, -)
freezeParamsExpo3=$(echo "$freezeParamsExpo3" | paste -sd, -)

cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20
# ****************************************
#
# PDF 0
# Generate Toys based on pdf 0
# # ****************************************


combine -M MultiDimFit \
  -d datacardMulti11.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_11_2016_13TeV=0 \
  -n fixed_pdfCat11_0 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsBern4" 
# ****************************************
# # PDF 1
# # ****************************************

  # Fit based on pdf 1
combine -M MultiDimFit \
  -d datacardMulti11.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_11_2016_13TeV=1 \
  -n fixed_pdfCat11_1 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsBern5" 
  
combine -M MultiDimFit \
  -d datacardMulti11.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-30,30 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_11_2016_13TeV=2 \
  -n fixed_pdfCat11_2 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  --expectSignal 0 \
  --freezeParameters "$freezeParamsExpo3" 