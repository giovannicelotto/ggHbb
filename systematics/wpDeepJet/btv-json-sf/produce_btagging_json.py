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

from convert_shapeSF import produce_reshape_json
from convert_wpSF    import produce_wp_json
from working_points  import produce_btagging_wp
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
naming_fixedWPb = {
    "2018": "106XUL18SF",
    "2017": "106XUL17SF",
    "2016preVFP": "106XUL16APVSF",
    "2016postVFP": "106XUL16SF",
    }
naming_fixedWPlight = {
    "2018": "UL2018",
    "2017": "UL2017",
    "2016preVFP": "UL2016_PreVFP",
    "2016postVFP": "UL2016_PostVFP",
    }
naming_itFit = {
    "2018": "UL2018",
    "2017": "UL2017",
    "2016preVFP": "UL2016",
    "2016postVFP": "UL2016",
    }

# need to read six files per tagger
# default file      -> btagging_fixedWP_SFb/TAGGER_TAG.csv
# year correlations -> btagging_fixedWP_SFb/TAGGER_TAG_YearCorrelation-V1.csv
# sfb sys breakdown -> btagging_fixedWP_SFb/TAGGER_TAG_CategoryBreakdown.csv
# sflight           -> btagging_fixedWP_SFlight/TAG_TAGGER.csv
# itfit             -> btagging_iterativeFit_SFb/TAGGER_TAG.csv # v2 for 2016
# itfit reduced jes -> btagging_iterativeFit_SFb/TAGGER_TAG_JESreduced.csv # v2 for 2016
def get_inputs(input_dir, tagger, year):
    taggerName = tagger.replace("deep","Deep")

    # fixedWP SFb files
    df_default = pd.read_csv(
        os.path.join(input_dir, "btagging_fixedWP_SFb", 
            taggerName+"_"+naming_fixedWPb[year]+".csv"),
        skipinitialspace = True
        )
    df_default = common.rename_columns(df_default, tagger)
    df_yc = pd.read_csv(
        os.path.join(input_dir, "btagging_fixedWP_SFb", 
            taggerName+"_"+naming_fixedWPb[year]+"_YearCorrelation-V1.csv"),
        skipinitialspace = True
        )
    df_yc = common.rename_columns(df_yc, tagger)
    df_cb = pd.read_csv(
        os.path.join(input_dir, "btagging_fixedWP_SFb", 
            taggerName+"_"+naming_fixedWPb[year]+"_CategoryBreakdown.csv"),
        skipinitialspace = True
        )
    df_cb = common.rename_columns(df_cb, tagger)

    # fixedWP SFlight file
    df_light = pd.read_csv(
        os.path.join(input_dir, "btagging_fixedWP_SFlight", 
            naming_fixedWPlight[year]+"_"+taggerName.replace("Jet", "Flavour")+".csv"),
        skipinitialspace = True
        )
    df_light = common.rename_columns(df_light, tagger)

    # itFit files
    suffix = "v3"

    df_itFit = pd.read_csv(
        os.path.join(input_dir, "btagging_iterativeFit_SFb",
            taggerName+"_"+naming_itFit[year]+suffix+".csv"),
        skipinitialspace = True
        )
    # dont need to rename default 2017/18 files as they are already renames
    df_itFit = common.rename_columns(df_itFit, tagger)
        
    df_itFit_jes = pd.read_csv(
        os.path.join(input_dir, "btagging_iterativeFit_SFb",
            taggerName+"_"+naming_itFit[year]+"_JESreduced"+suffix+".csv"),
        skipinitialspace = True
        )
    df_itFit_jes = common.rename_columns(df_itFit_jes, tagger)

    df_itFit.rename(columns={"formula ": "formula"}, inplace=True)
    df_itFit_jes.rename(columns={"formula ": "formula"}, inplace=True)


    # arrange fixedWP files
    # take uncertainty sources and central prediciton from breakdown file
    #df_cb = df_cb[~df_cb["sysType"].isin(["up", "down"])]
    # only take up/down from default file
    #df_default = df_default[~df_default["sysType"].isin(["central"])]
    # only take up/down_(un)correlated from yearcorrelation file
    df_yc = df_yc[~df_yc["sysType"].isin(["central", "up", "down"])]
    # concatenate to SFb
    #df_fixedWPb = pd.concat([df_cb, df_default, df_yc])
    df_fixedWPb = pd.concat([df_cb, df_yc])

    # split in comb and mujets
    df_comb   = df_fixedWPb[df_fixedWPb["measurementType"].isin(["comb"])]
    df_mujets = df_fixedWPb[df_fixedWPb["measurementType"].isin(["mujets"])]

    # get the json tables 
    corr_light =  produce_wp_json(df_light,  tagger, year)
    corr_comb  =  produce_wp_json(df_comb,   tagger, year)
    corr_mujets = produce_wp_json(df_mujets, tagger, year)
    
    # merge wp csv file
    df_fixedWP = pd.concat([df_light, df_fixedWPb])


    # arrange shape correction files
    # remove common systematics from jesreduced file
    jesSys = df_itFit_jes["sysType"].unique()
    allSys = df_itFit["sysType"].unique()
    commonSys = [x for x in allSys if x in jesSys]
    print("removing the following systematics from jes file:")
    print(commonSys)
    df_itFit_jes = df_itFit_jes[~df_itFit_jes["sysType"].isin(commonSys)]
    print("remaining: {}".format(df_itFit_jes["sysType"].unique()))
    # merge files
    df_itFit = pd.concat([df_itFit, df_itFit_jes])

    # get the json tables
    corr_itFit = produce_reshape_json(df_itFit, tagger, year)

    return corr_light, corr_comb, corr_mujets, df_fixedWP, corr_itFit, df_itFit

corr_deepJet_light, corr_deepJet_comb, corr_deepJet_mujets, df_deepJet_fixedWP, corr_deepJet_itFit, df_deepJet_itFit = get_inputs(input_dir, "deepJet", opts.year)
corr_deepCSV_light, corr_deepCSV_comb, corr_deepCSV_mujets, df_deepCSV_fixedWP, corr_deepCSV_itFit, df_deepCSV_itFit = get_inputs(input_dir, "deepCSV", opts.year)


# write shape correction stuff for testing
cset = CorrectionSet.model_validate(
    {
        "schema_version": VERSION,
        "corrections": [
            corr_deepCSV_itFit,
            corr_deepJet_itFit,
        ],
    }
)

# write output csv file
common.write_csv(
    os.path.join(outdir, "reshaping_deepCSV.csv"),
    df_deepCSV_itFit
    )
common.write_csv(
    os.path.join(outdir, "reshaping_deepJet.csv"),
    df_deepJet_itFit
    )

# write fixedWP correction stuff for testing
# ## Merging and evaluating
cset = CorrectionSet.model_validate(
    {
        "schema_version": VERSION,
        "corrections": [
            corr_deepCSV_light,
            corr_deepCSV_mujets,
            corr_deepCSV_comb,
            corr_deepJet_light,
            corr_deepJet_mujets,
            corr_deepJet_comb,
        ],
    }
)

# write csv output file
common.write_csv(
    os.path.join(outdir, "wp_deepCSV.csv"),
    df_deepCSV_fixedWP
    )
common.write_csv(
    os.path.join(outdir, "wp_deepJet.csv"),
    df_deepJet_fixedWP
    )


# add working points
corr_btagging_wp_deepJet = produce_btagging_wp("deepJet", opts.year)
corr_btagging_wp_deepCSV = produce_btagging_wp("deepCSV", opts.year)

# ## Final merging and evaluation
description = "This json file contains the corrections for deepJet and deepCSV AK4 taggers. "+\
"Corrections are supplied for b-tag discriminator shape corrections (shape) "+\
"and working point corrections (comb/mujets/incl). "+\
"For the working point corrections the SFs in 'mujets' and 'comb' are for b/c jets. "+\
"The 'mujets' SFs contain only corrections derived in QCD-enriched regions. "+\
"The 'comb' SFs contain corrections derived in QCD and ttbar-enriched regions. "+\
"Hence, 'comb' SFs can be used everywhere, except for ttbar-dileptonic enriched analysis regions. "+\
"For the ttbar-dileptonic regions the 'mujets' SFs should be used. "+\
"The 'incl' correction is for light-flavoured jets."

cset = CorrectionSet.model_validate(
    {
        "schema_version": VERSION,
        "description": description,
        "corrections": [
            corr_btagging_wp_deepCSV,
            corr_deepCSV_light,
            corr_deepCSV_mujets,
            corr_deepCSV_comb,
            corr_deepCSV_itFit,
            corr_btagging_wp_deepJet,
            corr_deepJet_light,
            corr_deepJet_mujets,
            corr_deepJet_comb,
            corr_deepJet_itFit,
        ],
    }
)

common.write_file(
    os.path.join(outdir, "btagging.json"),
    cset.model_dump_json(exclude_unset = True, indent=2)
    )


