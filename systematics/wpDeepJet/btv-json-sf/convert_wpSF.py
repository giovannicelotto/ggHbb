############################################
# based on ipynb provided by Xavier Coubez #
############################################

import pandas as pd
import re
import numpy as np
from correctionlib import convert
from correctionlib.schemav2 import (
    VERSION,
    Binning,
    Category,
    Correction,
    CorrectionSet,
    Formula,
    FormulaRef,
)
def build_wp_formulas(year):
    formulas = []
    formulas.append(Formula(
        nodetype   = "formula",
        expression = "[0]+[1]*log(x+19)*log(x+18)*(3+[2]*log(x+18))+[3]",
        parser     = "TFormula",
        variables  = ["pt"]
        ))
    formulas.append(Formula(
        nodetype   = "formula",
        expression = "[0]*(1.+[1]*x)/(1.+[2]*x)+[3]",
        parser     = "TFormula",
        variables  = ["pt"]
        ))
        
    return formulas


param = "(?:\+|\-|)\d+(?:\.\d*|)(?:e-?\d+|)"

#var = "(:?\)\*\(1(?P<s>\+|\-)\((?P<a>{param})(?:\+(?P<b>{param})\*x)?(?:\+(?P<c>{param})\*x\*x)?\)|$)".format(param=param)
#f1  = f"(?P<d>{param})\+(?P<e>{param})\*x\+(?P<f>{param})\*x\*x\+(?P<g>{param})\/x"+var
#f2  = f"(?P<d>{param})\+(?P<e>{param})\/sqrt\(x\)"+var
f3  = f"(?P<a>{param})\+\(*(?P<b>{param})\*\(?log\(x\+19\)\*\(?log\(x\+18\)\*\(3\(?(?P<c>{param})\*log\(x\+18\)+(?P<d>{param})?"
f4  = f"(?P<a>{param})\*\(1\.\+(?P<b>{param})\*x\)\/\(1\.\+(?P<c>{param})\*x\)(?P<d>{param})?"
#print(f1)
#print(f2)
print(f3)
print(f4)
#f1 = re.compile(f1)
#f2 = re.compile(f2)
f3 = re.compile(f3)
f4 = re.compile(f4)

def parse_formula(value):
    value = value.replace(" ","")
    value = value.replace("--","+")
    #print(f"parsing {value}")
    # try 2017 2018 formula
    match = f3.search(value)
    if not match is None:
        index = 0
    else:
        # try other formula
        match = f4.search(value)
        if match is None:
            raise ValueError(value)
        index = 1

    parameters=[float(match.group("a")), float(match.group("b")), float(match.group("c"))]
    d = 0
    if not match.group("d") is None:
        d = match.group("d")
    parameters.append(float(d))
    #print(parameters)
    return parameters, index
    

# ## Creating function to produce working point scale factors
def build_formula(sf):
    if len(sf) != 1:
        raise ValueError(sf)

    # get measurement type
    mtype = list(sf["measurementType"].unique())
    if len(mtype) != 1:
        raise ValueError(sf)
    mtype = mtype[0]

    value = sf.iloc[0]["formula"]
    #print(f'build_formula: {value}')
    if "x" in str(value):
        try: 
            # formularef for SFb
            parameters, index = parse_formula(value)
            return FormulaRef.parse_obj({
                "nodetype": "formularef",
                "index": index,
                "parameters": parameters
                })
        except:
            # formula for SFlight (appears to be changing all the time)
            return Formula.parse_obj({
                "nodetype": "formula",
                "expression": value.replace(" ",""),
                "parser": "TFormula",
                # For this case, since this is a working point SF, we know the parameter is the pT
                "variables": ["pt"],
                "parameters": [],
                })
    else:
        return float(value)
    
def build_ptbinning(sf):
    edges = sorted(set(sf["ptMin"]) | set(sf["ptMax"]))
    #print(f'build_ptbinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "pt",
            "edges": edges,
            "content": [
                build_formula(sf[(sf["ptMin"] >= lo) & (sf["ptMax"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ],
            "flow": "clamp",
        }
    )

def build_etabinning(sf):
    edges = sorted(set(sf["etaMin"]) | set(sf["etaMax"]))
    #print(f'build_etabinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "abseta",
            "edges": edges,
            "content": [
                build_ptbinning(sf[(sf["etaMin"] >= lo) & (sf["etaMax"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ],
            "flow": "error",
        }
    )

def build_flavor(sf):
    keys = list(map(int, sorted(sf["jetFlavor"].unique()))) 
    #print(f'build_flavor: {keys}, ')
    return Category.parse_obj(
        {
            "nodetype": "category",
            "input": "flavor",
            "content": [
                {"key": key, "value": build_etabinning(sf[sf["jetFlavor"] == key])}
                for key in keys
            ],
        }
    )

def build_wp(sf):
    keys = list(sf["OperatingPoint"].unique())
    #print(f'build_wp: {keys}')
    return Category.parse_obj(
        {
            "nodetype": "category",
            "input": "working_point",
            "content": [
                {"key": key, "value": build_flavor(sf[sf["OperatingPoint"] == key])}
                for key in keys
            ],
        }
    )


def build_systs(sf):
    keys = list(sf["sysType"].unique())
    #print(f'build_systs: {keys}')
    return Category.parse_obj(
        {
            "nodetype": "category",
            "input": "systematic",
            "content": [
                {"key": key, "value": build_wp(sf[sf["sysType"] == key])}
                for key in keys
            ],
        }
    )


description = "For the working point correction multiple different uncertainty schemes are provided. "+\
"If only one year is analyzed, the 'up' and 'down' systematics can be used. "+\
"If the fullRunII data is analyzed, 'up/down_correlated' and 'up/down_uncorrelated' "+\
"systematics are provided to be used instead of the 'up/down' ones, "+\
"which are supposed to be correlated/decorrelated between the different data years. "
descr_comb = "If the impact of b-tagging in the analysis is very dominant a further breakdown of "+\
"uncertainties is provided. These broken-down sources consist of "+\
"'up/down_isr/fsr/hdamp/jes/jer/pileup/qcdscale/statistic/topmass/type3'. "+\
"All of the sources can be correlated between the years, except the 'statistic' "+\
"source which is to be decorrelated between the years."
descr_mujets = "If the impact of b-tagging in the analysis is very dominant a further breakdown of "+\
"uncertainties is provided. These broken-down sources consist of "+\
"'up/down_jes/pileup/statistic/type3'. "+\
"All of the sources can be correlated between the years, except the 'statistic' "+\
"source which is to be decorrelated between the years."


def produce_wp_json(df, tagger, year):
    print(df)
    # get measurement type
    mtype = list(df["measurementType"].unique())
    if len(mtype) != 1:
        raise ValueError(df)
    mtype = mtype[0]

    # figure out description details based on measurement type
    if mtype == "incl":
        flav = "light"
        descr = description
    else:
        flav = "b/c"
        if mtype == "comb":
            descr = description+descr_comb
        elif mtype == "mujets":
            descr = description+descr_mujets
        else:
            descr = description

    correction = {
        "version": 1,
        "name": f"{tagger}_{mtype}",
        "description": f"{tagger} {mtype} working point scale factors for UL {year} {flav} jets. "+descr,
        "inputs": [
            {
                "name": "systematic", 
                "type": "string"
            },
            {
                "name": "working_point", 
                "type": "string",
                "description": "L/M/T"
            },
            {
                "name": "flavor",
                "type": "int",
                "description": "hadron flavor definition: 5=b, 4=c, 0=udsg"
            },
            {
                "name": "abseta",
                "type": "real"
            },
            {
                "name": "pt", 
                "type": "real"
            },
        ],
        "output": {"name": "weight", "type": "real"},
        "data": build_systs(df.copy()),
    }
    if mtype in ["comb", "mujets"]:
        correction["generic_formulas"] = build_wp_formulas(year)
    wp_json_sf = Correction.parse_obj(correction)
    return wp_json_sf

if __name__ == "__main__":
    import os
    import optparse
    import common
    parser = optparse.OptionParser()
    parser.add_option("-y","--year", dest = "year",
        help = "data era")
    parser.add_option("-b","--inFileB", dest = "inputFileB",
        help = "csv input file for b jet corrections")
    parser.add_option("-l","--inFileL", dest = "inputFileL",
        help = "csv input file for light jet corrections")
    parser.add_option("-t", "--tagger", dest = "taggerName",
        choices = ["deepCSV", "deepJet"],
        help = "name of tagger")
    (opts, args) = parser.parse_args()

    if not os.path.exists(opts.inputFileB):
        sys.exit(f"input file {opts.inputFileB} doesnt exist")
    if not os.path.exists(opts.inputFileL):
        sys.exit(f"input file {opts.inputFileL} doesnt exist")

    dfb = pd.read_csv(opts.inputFileB,
        skipinitialspace = True)
    dfl = pd.read_csv(opts.inputFileL,
        skipinitialspace = True)
    dfb = common.rename_columns(dfb, opts.taggerName)
    dfl = common.rename_columns(dfl, opts.taggerName)
    dfl = dfl[dfl.measurementType == "incl"]
    dfb_comb = dfb[dfb.measurementType == "comb"]
    dfb_muj  = dfb[dfb.measurementType == "mujets"]
    sfb_comb = produce_wp_json(dfb_comb, opts.taggerName, opts.year)
    sfb_muj = produce_wp_json(dfb_muj, opts.taggerName, opts.year)
    sfl = produce_wp_json(dfl, opts.taggerName, opts.year)

    cset = CorrectionSet.parse_obj(
        {
            "schema_version": VERSION,
            "corrections": [sfb_comb, sfb_muj, sfl],
        }
    )

    common.write_file(
        opts.inputFileB.replace(".csv", ".json"),
        cset.json(exclude_unset = True, indent=2)
        )
                                             
