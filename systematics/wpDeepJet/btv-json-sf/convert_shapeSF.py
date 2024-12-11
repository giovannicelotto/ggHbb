############################################
# based on ipynb provided by Xavier Coubez #
############################################

import pandas as pd
import re

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

def build_shape_formulas():
    formulas = []
    formulas.append(Formula(
        nodetype   = "formula",
        expression = "[1]*(x+[0])*(x+[0])*(x+[0])+[2]*(x+[0])*(x+[0])+[3]*(x+[0])+[4]",
        parser     = "TFormula",
        variables  = ["discriminant"]
        ))
    return formulas


#param = "(?:\+|\-|)\d+(?:\.\d*|)"
param = "(?:\+|\-|)\d+(?:\.\d*|)(?:e-?\d+|)"
f1 = f"(?P<a>{param})(?:\*\(x(?P<b>{param})\)){{3}}(?P<c>{param})(?:\*\(x(?P<d>{param})\)){{2}}(?P<e>{param})(?:\*\(x(?P<f>{param})\))(?P<g>{param})"
#print(f1)
f1 = re.compile(f1)

def parse_formula(value):
    #print(f"parsing {value}")
    match = f1.search(value)
    if match is None:
        raise ValueError(value)
    if not (match.group("b") == match.group("d") and match.group("b") == match.group("f")):
        raise ValueError(value)
    parameters=[
        float(match.group("b")), 
        float(match.group("a")), 
        float(match.group("c")), 
        float(match.group("e")), 
        float(match.group("g"))]
    #print(parameters)
    return parameters

def build_formula(sf):
    if len(sf) != 1:
        raise ValueError(sf)

    value = sf.iloc[0]["formula"]
    if "x" in value:
        parameters = parse_formula(value)
        return FormulaRef.parse_obj({
            "nodetype": "formularef",
            "index": 0,
            "parameters": parameters
            })
    else:
        return float(value)


def build_discrbinning(sf):
    edges = sorted(set(sf["discrMin"]) | set(sf["discrMax"]))
    #print(f'build_discrbinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "discriminant",
            "edges": edges,
            "content": [
                build_formula(sf[(sf["discrMin"] >= lo) & (sf["discrMax"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ],
            "flow": "clamp",
        }
    )

def build_ptbinning(sf):
    edges = sorted(set(sf["ptMin"]) | set(sf["ptMax"]))
    #print(f'build_ptbinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "pt",
            "edges": edges,
            "content": [
                build_discrbinning(sf[(sf["ptMin"] >= lo) & (sf["ptMax"] <= hi)])
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
    # above line needed otherwise pedantic complains about int not being int
    #print(f'build_flavor: {keys}, {type(keys[0])}')
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

def build_systs(sf):
    keys = list(sf["sysType"].unique())
    #print(f'build_systs: {keys}')
    return Category.parse_obj(
        {
            "nodetype": "category",
            "input": "systematic",
            "content": [
                {"key": key, "value": build_flavor(sf[sf["sysType"] == key])}
                for key in keys
            ],
        }
    )


description = "The scale factors have 8 default uncertainty sources (hf,lf,hfstats1/2,lfstats1/2,cferr1/2). "+\
"All except the cferr1/2 uncertainties are to be applied to light and b jets. "+\
"The cferr1/2 uncertainties are to be applied to c jets. "+\
"hf/lfstats1/2 uncertainties are to be decorrelated between years, the others correlated. "+\
"Additional jes-varied scale factors are supplied to be applied for the jes variations."

def produce_reshape_json(df, tagger, year):
    shape_json_sf = Correction.parse_obj(
        {
            "version": 3,
            "name": f"{tagger}_shape",
            "description": f"{tagger} reshaping scale factors for UL {year}. "+description,
            "inputs": [
                {
                    "name": "systematic", 
                    "type": "string"
                },
                {
                    "name": "flavor",
                    "type": "int",
                    "description": "hadron flavor definition: 5=b, 4=c, 0=udsg",
                },
                {
                    "name": "abseta", 
                    "type": "real"
                },
                {
                    "name": "pt", 
                    "type": "real"
                },
                {
                    "name": "discriminant",
                    "type": "real",
                    "description": f"{tagger} output value",
                },
            ],
            "output": {"name": "weight", "type": "real"},
            "generic_formulas": build_shape_formulas(),
            "data": build_systs(df.copy()),
        }
    )
    return shape_json_sf

if __name__ == "__main__":
    import os
    import optparse
    import common
    parser = optparse.OptionParser()
    parser.add_option("-y","--year", dest = "year", 
        help = "data era")
    parser.add_option("-i","--inFile", dest = "inputFile",
        help = "csv input file")
    parser.add_option("-t", "--tagger", dest = "taggerName",
        choices = ["deepCSV", "deepJet"],
        help = "name of tagger")
    (opts, args) = parser.parse_args()

    if not os.path.exists(opts.inputFile):
        sys.exit(f"input file {opts.inputFile} doesnt exist")

    df = pd.read_csv(opts.inputFile,
        skipinitialspace = True)
    df = common.rename_columns(df, opts.taggerName)
    print(df)
    print(df.columns)
    df = df[df["OperatingPoint"] == "shape"]
    sf = produce_reshape_json(df, opts.taggerName, opts.year)

    cset = CorrectionSet.parse_obj(
        {
            "schema_version": VERSION,
            "corrections": [sf],
        }
    )

    common.write_file(
        opts.inputFile.replace(".csv", ".json"),
        cset.json(exclude_unset = True, indent=2)
        )
