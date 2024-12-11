############################################
# based on ipynb provided by Xavier Coubez #
############################################

import pandas as pd
import os
import sys
import re
def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)
    print("output file {} written".format(path))

def write_csv(path, df):
    df.to_csv(path, index=False)
    print("output file {} written".format(path))

def rename_columns(df, tagger, mtype=None):
    fnTagger = tagger.replace("deep","Deep")
    df.rename(columns={f"{fnTagger};OperatingPoint":"OperatingPoint"},  inplace=True, errors="ignore")
    df.rename(columns={f"{fnTagger}; OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f"{tagger};OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f"{tagger.lower()};OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f";OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f"DeepFlavour;OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f"DeepFlavourC;OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={f"{fnTagger}C;OperatingPoint":"OperatingPoint"},  inplace=True, errors="ignore")
    df.rename(columns={f"{fnTagger}C;OperatingPoint":"OperatingPoint"},  inplace=True, errors="ignore")
    if not mtype is None:
        df.rename(columns={f"{mtype};OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
        df.rename(columns={f"{mtype.lower()};OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
        df.rename(columns={f"{mtype.lower()} ;OperatingPoint":"OperatingPoint"}, inplace=True, errors="ignore")
    df.rename(columns={"jetFlavour": "jetFlavor"}, inplace=True)
    df.rename(columns={"formula ": "formula"}, inplace=True)
    df.rename(columns={"ptmax": "ptMax"}, inplace=True)
    df.replace({"jetFlavor": {0: 5, 1: 4, 2: 0}}, inplace=True)
    df.replace({"etaMax": {2.4: 2.5}}, inplace=True)
    df.replace({"etaMin": {-2.5: 0., -2.4: 0.}}, inplace=True)
    df.replace({"OperatingPoint": {0: "L", 1: "M", 2: "T", 3: "shape"}}, inplace=True)
    return df
