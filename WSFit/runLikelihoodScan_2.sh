#!/bin/bash

cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv
freezeParamsBern2=$(
  cat <<EOF
env_pdf_Exponential_1_cat2_exp1_p1
env_pdf_Exponential_1_cat2_turnon_cutoff
env_pdf_Exponential_1_cat2_turnon_beta
env_pdf_Exponential_1_cat2_z_norm
env_pdf_PowerLaw_1_cat2_pow1_p1
env_pdf_PowerLaw_1_cat2_turnon_cutoff
env_pdf_PowerLaw_1_cat2_turnon_beta
env_pdf_PowerLaw_1_cat2_z_norm
shapeSig_signal_Cat2__norm
EOF
)
freezeParamsPow1=$(
  cat <<EOF
env_pdf_Bernstein_2_cat2_bern2_p0
env_pdf_Bernstein_2_cat2_bern2_p1
env_pdf_Bernstein_2_cat2_z_norm
env_pdf_Bernstein_2_cat2_turnon_beta
env_pdf_Bernstein_2_cat2_turnon_cutoff
env_pdf_Exponential_1_cat2_exp1_p1
env_pdf_Exponential_1_cat2_turnon_cutoff
env_pdf_Exponential_1_cat2_turnon_beta
env_pdf_Exponential_1_cat2_z_norm
shapeSig_signal_Cat2__norm
EOF
)

freezeParamsExpo1=$(
  cat <<EOF
env_pdf_Bernstein_2_cat2_bern2_p0
env_pdf_Bernstein_2_cat2_bern2_p1
env_pdf_Bernstein_2_cat2_z_norm
env_pdf_Bernstein_2_cat2_turnon_beta
env_pdf_Bernstein_2_cat2_turnon_cutoff
env_pdf_PowerLaw_1_cat2_pow1_p1
env_pdf_PowerLaw_1_cat2_turnon_cutoff
env_pdf_PowerLaw_1_cat2_turnon_beta
env_pdf_PowerLaw_1_cat2_z_norm
shapeSig_signal_Cat2__norm
EOF
)



# unisce le righe con virgole
freezeParamsBern2=$(echo "$freezeParamsBern2" | paste -sd, -)
freezeParamsExpo1=$(echo "$freezeParamsExpo1" | paste -sd, -)
freezeParamsPow1=$(echo "$freezeParamsPow1" | paste -sd, -)

cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20

combine \
  -M MultiDimFit \
  -d datacardMulti2.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_2_2016_13TeV=0 \
  -n fixed_pdfCat2_0 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsBern2" -v3






combine \
  -M MultiDimFit \
  -d datacardMulti2.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_2_2016_13TeV=2 \
  -n fixed_pdfCat2_2 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsExpo1" 

combine -M MultiDimFit \
	-d datacardMulti2.txt \
	--algo grid \
	--setParameterRanges r=-3,5 \
	--points 20 \
	--cminDefaultMinimizerStrategy 0\
	--saveNLL -n Envelope -m 125  \
	--freezeParameters shapeSig_signal_Cat2__norm
	-t -1 --expectSignal 1

combine \
  -M MultiDimFit \
  -d datacardMulti2.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_2_2016_13TeV=1 \
  -n fixed_pdfCat2_1 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsPow1" 



