#!/bin/bash
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
freezeParamsBern3=$(
  cat <<EOF
env_pdf_Exponential_3_cat1_exp3_p1
env_pdf_Exponential_3_cat1_exp3_p2
env_pdf_Exponential_3_cat1_exp3_f1
env_pdf_Exponential_3_cat1_z_norm
env_pdf_PowerLaw_1_cat1_pow1_p1
env_pdf_PowerLaw_1_cat1_z_norm
env_pdf_Bernstein_4_cat1_bern4_p0
env_pdf_Bernstein_4_cat1_bern4_p1
env_pdf_Bernstein_4_cat1_bern4_p2
env_pdf_Bernstein_4_cat1_bern4_p3
env_pdf_Bernstein_4_cat1_z_norm
shapeSig_signal_Cat1__norm
pdfindex_1_2016_13TeV
EOF
)
freezeParamsBern4=$(
  cat <<EOF
env_pdf_Exponential_3_cat1_exp3_p1
env_pdf_Exponential_3_cat1_exp3_p2
env_pdf_Exponential_3_cat1_exp3_f1
env_pdf_Exponential_3_cat1_z_norm
env_pdf_PowerLaw_1_cat1_pow1_p1
env_pdf_PowerLaw_1_cat1_z_norm
env_pdf_Bernstein_3_cat1_bern3_p0
env_pdf_Bernstein_3_cat1_bern3_p1
env_pdf_Bernstein_3_cat1_bern3_p2
env_pdf_Bernstein_3_cat1_z_norm
shapeSig_signal_Cat1__norm
pdfindex_1_2016_13TeV
EOF
)
freezeParamsPow1=$(
  cat <<EOF
env_pdf_Exponential_3_cat1_exp3_p1
env_pdf_Exponential_3_cat1_exp3_p2
env_pdf_Exponential_3_cat1_exp3_f1
env_pdf_Exponential_3_cat1_z_norm
env_pdf_Bernstein_3_cat1_bern3_p0
env_pdf_Bernstein_3_cat1_bern3_p1
env_pdf_Bernstein_3_cat1_bern3_p2
env_pdf_Bernstein_3_cat1_z_norm
env_pdf_Bernstein_4_cat1_bern4_p0
env_pdf_Bernstein_4_cat1_bern4_p1
env_pdf_Bernstein_4_cat1_bern4_p2
env_pdf_Bernstein_4_cat1_bern4_p3
env_pdf_Bernstein_4_cat1_z_norm
shapeSig_signal_Cat1__norm
pdfindex_1_2016_13TeV
EOF
)
freezeParamsExpo3=$(
  cat <<EOF
env_pdf_PowerLaw_1_cat1_pow1_p1
env_pdf_PowerLaw_1_cat1_z_norm
env_pdf_Bernstein_3_cat1_bern3_p0
env_pdf_Bernstein_3_cat1_bern3_p1
env_pdf_Bernstein_3_cat1_bern3_p2
env_pdf_Bernstein_3_cat1_z_norm
env_pdf_Bernstein_4_cat1_bern4_p0
env_pdf_Bernstein_4_cat1_bern4_p1
env_pdf_Bernstein_4_cat1_bern4_p2
env_pdf_Bernstein_4_cat1_bern4_p3
env_pdf_Bernstein_4_cat1_z_norm
shapeSig_signal_Cat1__norm
pdfindex_1_2016_13TeV
EOF
)

  
  


freezeParamsBern3=$(echo "$freezeParamsBern3" | paste -sd, -)
freezeParamsBern4=$(echo "$freezeParamsBern4" | paste -sd, -)
freezeParamsPow1=$(echo "$freezeParamsPow1" | paste -sd, -)
freezeParamsExpo3=$(echo "$freezeParamsExpo3" | paste -sd, -)
echo "freezeParamsBern3=$freezeParamsBern3"

cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20
# ****************************************
#
# PDF 0
# Generate Toys based on pdf 0
# # ****************************************
combine -M MultiDimFit \
  -d datacardMulti1.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_1_2016_13TeV=0 \
  -n fixed_pdfCat1_0 \
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsBern3"\
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1
# ****************************************
# # PDF 1
# # ****************************************

  # Fit based on pdf 1
combine -M MultiDimFit \
  -d datacardMulti1.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_1_2016_13TeV=1 \
  -n fixed_pdfCat1_1 \
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsBern4"\
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1
  
  
  # ****************************************
  # # PDF 2
  # # ****************************************
  
# Fit based on pdf 2

combine -M MultiDimFit \
  -d datacardMulti1.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_1_2016_13TeV=2 \
  -n fixed_pdfCat1_2 \
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsPow1"\
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1




combine -M MultiDimFit \
  -d datacardMulti1.txt \
  --algo grid \
  --points 20 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  -m 125 \
  --setParameters pdfindex_1_2016_13TeV=3 \
  -n fixed_pdfCat1_3 \
  --expectSignal 1 \
  -t -1 \
  --freezeParameters "$freezeParamsExpo3"\
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1