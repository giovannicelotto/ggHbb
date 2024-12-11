############################################
# based on ipynb provided by Xavier Coubez #
############################################

import pandas as pd
import os
import sys

from correctionlib import convert
from correctionlib.schemav2 import (
    VERSION,
    CorrectionSet,
)

from convert_subjetSF import produce_subjet_json
import common

import optparse
parser = optparse.OptionParser()
parser.add_option("--year","-y",dest="year",
    choices = ["2018", "2017", "2016preVFP", "2016postVFP"],
    help = "choose data taking year to convert files")

(opts, args) = parser.parse_args()

input_dir = os.path.abspath(
    os.path.join(".", "..", "btv-scale-factors", "UL"+opts.year, "Subjet_btagging"))
if not os.path.exists(input_dir):
    print(f"input directory {input_dir} does not exist")
    sys.exit()

# get output directory
outdir = os.path.join("data", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)


naming = {
    "2018": "106XUL18SF",
    "2017": "106XUL17SF",
    "2016preVFP": "106XUL16APVSF",
    "2016postVFP": "106XUL16SF",
    }
# need to read two files
# Subjet_btagging/DeepCSV_YEAR_Subjets_btag.csv
# Subjet_btagging/DeepCSV_YEAR_Subjets_mistag.csv
def get_inputs(input_dir, tagger, year):

    df_btag = pd.read_csv(
        os.path.join(input_dir, "DeepCSV_"+naming[year]+"_Subjets_btag.csv"),
        skipinitialspace = True
        )
    df_btag = common.rename_columns(df_btag, "deepCSV")
    df_mistag = pd.read_csv(
        os.path.join(input_dir, "DeepCSV_"+naming[year]+"_Subjets_mistag.csv"),
        skipinitialspace = True
        )
    df_mistag = common.rename_columns(df_mistag, tagger)

    # concatenate dataframes
    df = pd.concat([df_btag, df_mistag])
    
    # get correction
    correction = produce_subjet_json(df, tagger, year)
    return df, correction

    
# get json table
df_deepCSV, corr_deepCSV = get_inputs(input_dir, "deepCSV", opts.year)

# write output file
common.write_csv(
    os.path.join(outdir, "subjet_deepCSV.csv"),
    df_deepCSV
    )

# ## Final merging and evaluation
description = "This json file contains the corrections for deepCSV subjet tagging. "

cset = CorrectionSet.parse_obj(
    {
        "schema_version": VERSION,
        "description": description,
        "corrections": [
            corr_deepCSV
        ],
    }
)


common.write_file(
    os.path.join(outdir, "subjet_tagging.json"),
    cset.json(exclude_unset = True, indent=2)
    )


