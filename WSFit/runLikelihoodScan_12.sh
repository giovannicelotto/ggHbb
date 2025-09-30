#!/bin/bash
cd /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20
freezeParamsBern2=$(
  cat <<EOF
env_pdf_Exponential_1_cat12_exp1_p1
env_pdf_Exponential_1_cat12_z_norm
env_pdf_Bernstein_3_cat12_bern3_p0
env_pdf_Bernstein_3_cat12_bern3_p1
env_pdf_Bernstein_3_cat12_bern3_p2
env_pdf_Bernstein_3_cat12_z_norm
shapeSig_signal_Cat12__norm
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
EOF
)
# unisce le righe con virgole
freezeParamsBern2=$(echo "$freezeParamsBern2" | paste -sd, -)
freezeParamsBern3=$(echo "$freezeParamsBern3" | paste -sd, -)
freezeParamsExpo1=$(echo "$freezeParamsExpo1" | paste -sd, -)
combine \
  -M MultiDimFit \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_12_2016_13TeV=0 \
  -n fixed_pdfCat12_0 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsBern2" 






combine \
  -M MultiDimFit \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_12_2016_13TeV=1 \
  -n fixed_pdfCat12_1 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsBern3" 

combine \
  -M MultiDimFit \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1\
  -d datacardMulti12.txt \
  --algo grid \
  --points 50 \
  --setParameterRanges r=-3,5 \
  --saveNLL \
  --setParameters pdfindex_12_2016_13TeV=2 \
  -n fixed_pdfCat12_2 \
  -m 125 \
  -t -1 \
  --expectSignal 1 \
  --freezeParameters "$freezeParamsExpo1" 



combine -M MultiDimFit \
	-d datacardMulti12.txt \
	--algo grid \
	--setParameterRanges r=-3,5 \
	--points 20 \
	--cminDefaultMinimizerStrategy 0\
	--saveNLL -n Envelope -m 125  \
	--X-rtd REMOVE_CONSTANT_ZERO_POINT=1  \
	--freezeParameters shapeSig_signal_Cat12__norm \
	-t -1 --expectSignal 1