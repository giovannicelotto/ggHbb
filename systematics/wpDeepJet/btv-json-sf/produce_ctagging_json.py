############################################
# based on ipynb provided by Xavier Coubez #
############################################

import pandas as pd
import os
import sys

from correctionlib.schemav2 import (
    VERSION,
    CorrectionSet,
)

from convert_shapeSF_c import produce_reshape_json
from convert_wpSF_c    import produce_wp_json
from working_points    import produce_ctagging_wp
import common
import optparse
parser = optparse.OptionParser()
parser.add_option("--year","-y",dest="year",
    choices = ["2018", "2017", "2016preVFP", "2016postVFP"],
    help = "choose data taking year to convert files")

(opts, args) = parser.parse_args()

input_dir = os.path.abspath(
    os.path.join(".", "..", "btv-scale-factors", "UL"+opts.year))
if not os.path.exists(input_dir):
    print(f"input directory {input_dir} does not exist")
    sys.exit()

# get output directory
outdir = os.path.join("data", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)

# figure out naming scheme of csv files
naming_fixedWP = {
    "2018": "UL18",
    "2017": "UL17",
    "2016preVFP": "UL16preVFP",
    "2016postVFP": "UL16postVFP",
    }
naming_itFit = {
    "2018": "Summer20UL18",
    "2017": "Summer20UL17",
    "2016preVFP": "Summer20UL16PreVFP",
    "2016postVFP": "Summer20UL16PostVFP",
    }


# add working points
corr_ctagging_wp_deepJet = produce_ctagging_wp("deepJet", opts.year)
corr_ctagging_wp_deepCSV = produce_ctagging_wp("deepCSV", opts.year)

# need to read six files per tagger
# fixedWP SFc file  -> ctagging_fixedWP_SFc/TAGGERTAG.csv
# fixedWP SFl file  -> ctagging_fixedWP_SFlight/TAGGERTAG.csv
# fixedWP sfb file  -> ctagging_fixedWP_Sfb/TAGGER.csv
# itfit file        -> ctagging_reshaping/TAGGER_ctagSF_TAG_interp_withJEC.csv
#DeepCSV_ctagSF_Summer20UL17_interp.root
def get_inputs(input_dir, tagger, year):
    taggerName = tagger.replace("deep","Deep")

    # fixedWP SFc files
    df_wpc = pd.read_csv(
        os.path.join(input_dir, "ctagging_fixedWP_SFc",
            tagger+naming_fixedWP[year]+".csv"),
        skipinitialspace = True
        )
    df_wpc = common.rename_columns(df_wpc, tagger)
    # fixedWP SFl files
    df_wpl = pd.read_csv(
        os.path.join(input_dir, "ctagging_fixedWP_SFlight",
            tagger+naming_fixedWP[year]+".csv"),
        skipinitialspace = True
        )
    df_wpl = common.rename_columns(df_wpl, tagger)
    # fixedWP SFb files
    df_wpb = pd.read_csv(
        os.path.join(input_dir, "ctagging_fixedWP_SFb", 
            tagger+".csv"),
        skipinitialspace = True
        )
    df_wpb = common.rename_columns(df_wpb, tagger)


    # itFit files
    df_itFit = pd.read_csv(
        os.path.join(input_dir, "ctagging_reshaping",
            taggerName+"_ctagSF_"+naming_itFit[year]+"_interp_withJEC.csv"),
        skipinitialspace = True
        )

    # arrange fixedWP files
    def get_unc(x):
        if not ("+" in x or "-" in x):
            return 0.
        x = x.replace("-","+-")
        val, unc = x.split("+")
        return float(unc)
    def get_val(x):
        x = x.replace("-", "+-")
        return float(x.split("+")[0])    
    
    # split formula in SF and uncertainty
    df_wpc["sf"]  = df_wpc["formula"].apply(get_val)
    df_wpc["unc"] = df_wpc["formula"].apply(get_unc)
    df_wpc["formula"] = df_wpc["sf"]+df_wpc["unc"]
    df_wpc["measurementType"] = "wcharm"

    ## temporarily copy cSF for bSF
    #df_wpb = df_wpc.copy()
    ## double c uncertainty
    #df_wpb["unc"] = 2.*df_wpb["unc"]
    #df_wpb["formula"] = df_wpb["sf"]+df_wpb["unc"]
    #df_wpb["jetFlavor"] = 5

    # only add up/down variations for SFb. Split uncertainty sources not needed atm
    df_wpb = df_wpb[(df_wpb.sysType.isin(["up", "down", "central"]))]
    print(df_wpb)

    # merge wp csv file
    df_fixedWP = pd.concat([df_wpc, df_wpb, df_wpl])
    corr_fixedWP = produce_wp_json(df_fixedWP, tagger, year)

    # remove additional lines from df
    df_fixedWP.drop("unc", inplace=True, axis=1)
    df_fixedWP.drop("sf", inplace=True, axis=1)

    # get the json tables for it fit
    corr_itFit = produce_reshape_json(df_itFit, tagger, year)

    return corr_fixedWP, df_fixedWP, corr_itFit, df_itFit

corr_deepJet_fixedWP, df_deepJet_fixedWP, corr_deepJet_itFit, df_deepJet_itFit = get_inputs(input_dir, "deepJet", opts.year)
corr_deepCSV_fixedWP, df_deepCSV_fixedWP, corr_deepCSV_itFit, df_deepCSV_itFit = get_inputs(input_dir, "deepCSV", opts.year)

# write shape correction stuff for testing
cset = CorrectionSet.parse_obj(
    {
        "schema_version": VERSION,
        "corrections": [
            corr_deepCSV_itFit,
            corr_deepJet_itFit,
        ],
    }
)
print("itfit success")

# write output csv file
common.write_csv(
    os.path.join(outdir, "ctagging_reshaping_deepCSV.csv"),
    df_deepCSV_itFit
    )
common.write_csv(
    os.path.join(outdir, "ctagging_reshaping_deepJet.csv"),
    df_deepJet_itFit
    )

# write fixedWP correction stuff for testing
# ## Merging and evaluating
cset = CorrectionSet.parse_obj(
    {
        "schema_version": VERSION,
        "corrections": [
            corr_deepCSV_fixedWP,
            corr_deepJet_fixedWP,
        ],
    }
)
print("fixedwp success")

# write csv output file
common.write_csv(
    os.path.join(outdir, "ctagging_wp_deepCSV.csv"),
    df_deepCSV_fixedWP
    )
common.write_csv(
    os.path.join(outdir, "ctagging_wp_deepJet.csv"),
    df_deepJet_fixedWP
    )

# add working points
corr_ctagging_wp_deepJet = produce_ctagging_wp("deepJet", opts.year)
corr_ctagging_wp_deepCSV = produce_ctagging_wp("deepCSV", opts.year)

# ## Final merging and evaluation
description = "This json file contains the corrections for deepJet and deepCSV AK4 c-taggers. "+\
"Corrections are supplied for c-tagging working point corrections (wp) "+\
"and c-tagging discriminator shape corrections (shape) "

cset = CorrectionSet.parse_obj(
    {
        "schema_version": VERSION,
        "description": description,
        "corrections": [
            corr_ctagging_wp_deepJet,
            corr_deepJet_fixedWP,
            corr_deepJet_itFit,
            corr_ctagging_wp_deepCSV,
            corr_deepCSV_fixedWP,
            corr_deepCSV_itFit,
        ],
    }
)
print("full success")
common.write_file(
    os.path.join(outdir, "ctagging.json"),
    cset.json(exclude_unset = True, indent=2)
    )


