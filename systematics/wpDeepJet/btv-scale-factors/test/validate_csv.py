import pandas as pd
import sys
import os
import re
in_file = sys.argv[1]

error=False

# validate file naming scheme
file_name = os.path.basename(in_file)
dir_name = os.path.basename(os.path.dirname(in_file))
csv_name = os.path.basename(os.path.dirname(os.path.dirname(in_file)))
campaign_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(in_file))))

if not csv_name=="csv":
    # skip irrelevant files
    exit()

if campaign_name.startswith("UL"):
    # not doing any validation for the old Run2 files
    exit()
    
try:
    year, tag = campaign_name.split("_")
    intYear = int(year[:4])
except:
    print(f"Invalid campaign name of folder '{campaign_name}'")
    error=True
if not error:
    if intYear <= 2018:
        # not doing any validation for the old Run2 files
        exit()


_dir_regex="^(b|c)tagging_(fixedWP|shape)(_(SFb|SFc|SFlight|itFit))?$"
if not re.search(_dir_regex, dir_name):
    print(f"Directory name '{dir_name}' of file does not match, should be '{_dir_regex[1:-1]}'\\n")
    error=True

_file_regex="^(deepJet|particleNet|robustParticleTransformer)(_\w+)?_v\d+\.csv$"
if not re.match(_file_regex, file_name):
    print(f"File name '{file_name}' does not match expectations, should be '(deepJet|particleNet|robustParicleTransformer)_[descriptor]_v*.csv'.\\n")
    print("Descriptor should be for example the SF method (shape|tnp|sys8|SFcomb|...) or a descriptor of the uncertainties, e.g. (breakdown|JESreduced|...).\\n")
    error=True

# check for spaces in file
try:
    with open(in_file, "r") as f:
        lines = "".join(f.readlines())
    if " " in lines:
        print("csv file should not contain spaces between columns/commas/values/etc\\n")
        error=True
except:
    raise ValueError("csv file could not be read - contains illegal characters.\\n")

try:
    df = pd.read_csv(in_file)
except:
    raise ValueError("Cant read csv file.\\n")

# check if necessary columns are present
columns = df.columns.values

# check working point column
allowed_values = ["L", "M", "T", "XT", "XXT", "-"]
if not "wp" in columns:
    print(f"First column should be called 'wp', allowed values in that column are '{'/'.join(allowed_values)}'.\\n")
    error=True
else:
    for value in df["wp"].unique():
        if not value in allowed_values:
            print(f"Value '{value}' in 'wp' column not allowed.\\n")
            error=True

# DeepCSV;OperatingPoint, measurementType, sysType, jetFlavor, etaMin, etaMax, ptMin, ptMax, discrMin, discrMax, formula
# check type column
allowed_values = ["mujets", "qcd_tt2L", "qcd_tt1L", "comb", "light", "shape"]
allowed_values+= ["sys8","ptrel","ltsv","tnp","kinfit","wc","negtag"]
if not "type" in columns:
    print(f"Column 'type' does not exist.\\n")
    error=True
else:
    for value in df["type"].unique():
        if not value in allowed_values:
            print(f"Value '{value}' in 'type' column not allowed. Allowed values are '{'/'.join(allowed_values)}'.\\n")
            error=True

# check jet flavor column
allowed_values = [0, 4, 5]
if not "flav" in columns:
    print(f"Column 'flav' does not exist.\\n")
    error=True
else:
    for value in df["flav"].unique():
        if not value in allowed_values:
            print(f"Value '{value}' in 'flav' column not allowed. Allowed values are '{'/'.join(map(str, allowed_values))}'.\\n")
            error=True

# check syst column
if not "syst" in columns:
    print(f"Column 'syst' does not exist.\\n")
    error=True
else:
    for value in df["syst"].unique():
        if not (value=="central" or value.startswith("up") or value.startswith("down")):
            print(f"Value '{value}' in 'syst' column not allowed. Allowed values are either 'central' or 'up/down_*'.\\n")
            error=True

# check eta/pt/discr columns
col_names = ["etaMin", "etaMax", "ptMin", "ptMax", "discrMin", "discrMax"]
for c in col_names:
    if not c in columns:
        print(f"Column '{c}' does not exist.\\n")
        error=True
    else:
        values = [float(v) for v in df[c].unique()]
        if min(values) < 0.:
            print(f"Value '{min(values)}'<0 not allowed in '{c}' column.\\n")
            error=True


other_cols = ["formula"]
for c in other_cols:
    if not c in columns:
        print(f"Column '{c}' does not exist.\\n")
        error=True
