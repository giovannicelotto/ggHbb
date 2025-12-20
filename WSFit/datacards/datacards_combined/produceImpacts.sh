combineCards.py  datacardMulti4.txt datacardMulti14.txt > datacardMulti_4_14.txt
text2workspace.py datacardMulti_4_14.txt
combineTool.py -M Impacts -d datacardMulti_4_14.root -m 125  -n .impacts --setParameterRanges r=-9,11 --doInitialFit --robustFit 1 -t -1 --expectSignal 1 
combineTool.py -M Impacts -d datacardMulti_4_14.root -m 125  -n .impacts --setParameterRanges r=-9,11 --doFits --robustFit 1 -t -1 --expectSignal 1 
combineTool.py -M Impacts -d datacardMulti_4_14.root -m 125  -n .impacts --setParameterRanges r=-9,11 -o impacts_4_14.json
plotImpacts.py -i impacts_4_14.json -o impacts_4_14


#--freezeParameters pdfindex_2_2016_14TeV,env_pdf_Exponential_1_cat2_exp1_p1,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_Exponential_4_cat2_exp4_f1,env_pdf_Exponential_4_cat2_exp4_p1,env_pdf_Exponential_4_cat2_exp4_p2,env_pdf_Exponential_1_cat2_turnon_beta,env_pdf_Exponential_1_cat2_turnon_cutoff,env_pdf_PowerLaw_1_cat2_pow1_p1,env_pdf_PowerLaw_1_cat2_turnon_beta,env_pdf_PowerLaw_1_cat2_turnon_cutoff,env_pdf_Exponential_4_cat2_turnon_beta,env_pdf_Exponential_4_cat2_turnon_cutoff,env_pdf_Exponential_1_cat2_exp1_p1 --X-rtd MINIMIZER_freezeDisassociatedParams