combine -M MultiDimFit -d datacardMulti2.txt --algo grid  --points 50 --setParameterRanges r=-3,5  --saveNLL --setParameters pdfindex_2_2016_13TeV=0 -n fixed_pdfCat2_0 -m 125  -t -1 --expectSignal 1 --freezeParameters \
	env_pdf_Exponential_1_cat2_exp1_p1,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,\
	env_pdf_PowerLaw_1_cat2_pow1_p1,env_pdf_PowerLaw_1_cat2_turnon_cutoff,env_pdf_PowerLaw_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,env_pdf_PowerLaw_1_cat2_z_norm,\
       shapeSig_signal_Cat2__norm	-v3

#combine -M Significance -d datacardMulti2.txt --setParameters pdfindex_2_2016_13TeV=0 -n fixed_pdfCat2_0 -m 125  -t -1 --expectSignal 1 --freezeParameters env_pdf_Exponential_1_cat2_exp1_p1,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_PowerLaw_1_cat2_pow1_p1,env_pdf_PowerLaw_1_cat2_turnon_cutoff,env_pdf_PowerLaw_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,env_pdf_PowerLaw_1_cat2_z_norm,shapeSig_signal_Cat2__norm -v3

combine -M MultiDimFit -d datacardMulti2.txt --algo grid  --points 50 --setParameterRanges r=-3,5  --saveNLL --setParameters pdfindex_2_2016_13TeV=1 -n fixed_pdfCat2_1 -m 125  -t -1 --expectSignal 1 --freezeParameters \
	env_pdf_Exponential_1_cat2_exp1_p1,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,\
	env_pdf_Bernstein_2_cat2_bern2_p0,env_pdf_Bernstein_2_cat2_bern2_p1,env_pdf_Bernstein_2_cat2_bern2_p2,env_pdf_Bernstein_2_cat2_z_norm,env_pdf_Bernstein_2_cat2_turnon_beta,env_pdf_Bernstein_2_cat2_turnon_cutoff,\
	#env_pdf_PowerLaw_1_cat2_pow1_p1,env_pdf_PowerLaw_1_cat2_turnon_cutoff,env_pdf_PowerLaw_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,env_pdf_PowerLaw_1_cat2_z_norm,\
	shapeSig_signal_Cat2__norm -v3


combine -M MultiDimFit -d datacardMulti2.txt --algo grid  --points 50 --setParameterRanges r=-3,5  --saveNLL --setParameters pdfindex_2_2016_13TeV=2 -n fixed_pdfCat2_2 -m 125  -t -1 --expectSignal 1 --freezeParameters \
        #env_pdf_Exponential_1_cat2_exp1_p1,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,\
        env_pdf_Bernstein_2_cat2_bern2_p0,env_pdf_Bernstein_2_cat2_bern2_p1,env_pdf_Bernstein_2_cat2_bern2_p2,env_pdf_Bernstein_2_cat2_z_norm,env_pdf_Bernstein_2_cat2_turnon_beta,env_pdf_Bernstein_2_cat2_turnon_cutoff,\
        env_pdf_PowerLaw_1_cat2_pow1_p1,env_pdf_PowerLaw_1_cat2_turnon_cutoff,env_pdf_PowerLaw_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_z_norm,env_pdf_PowerLaw_1_cat2_z_norm,\
        shapeSig_signal_Cat2__norm -v3
