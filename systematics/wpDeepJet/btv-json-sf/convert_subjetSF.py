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
def build_subjet_formulas():
    formulas = []
    formulas.append(Formula(
        nodetype   = "formula",
        expression = "[0]+[1]*x+[2]*x*x+[3]*x*x*x+[4]*x*x*x*x",
        parser     = "TFormula",
        variables  = ["pt"]
        ))
    return formulas

param = "(?:\+|\-|)\d+(?:\.\d*|)(?:e-?\d+|)"
f1 = f"(?:(?P<a>{param}))(?:(?P<b>{param})\*x)(?:(?P<c>{param})\*x\*x)(?:(?P<d>{param})\*x\*x\*x)?(?:(?P<e>{param})\*x\*x\*x\*x)?"
print(f1)
f1 = re.compile(f1)

def parse_formula(value):
    print(f"parsing {value}")
    match = f1.search(value)
    if match is None:
        raise ValueError(value)
    parameters = [float(match.group("a")), float(match.group("b")), float(match.group("c"))]
    d = 0
    if not match.group("d") is None:
        d = match.group("d")
    parameters.append(float(d))
    e = 0
    if not match.group("e") is None:
        e = match.group("e")
    parameters.append(float(e))
    print(parameters)
    return parameters

# ## Creating function to produce working point scale factors
def build_formula(sf):
    if len(sf) != 1:
        raise ValueError(sf)

    value = sf.iloc[0]["formula"]
    if "x" in str(value):
        return FormulaRef.parse_obj({
            "nodetype": "formularef",
            "index": 0,
            "parameters": parse_formula(value),
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

def build_method(sf):
    keys = list(sf["measurementType"].unique())
    #print(f'build_method: {keys}')
    return Category.parse_obj(
        {
            "nodetype": "category",
            "input": "method",
            "content": [
                {"key": key, "value": build_wp(sf[sf["measurementType"] == key])}
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
                {"key": key, "value": build_method(sf[sf["sysType"] == key])}
                for key in keys
            ],
        }
    )

description = "The names of the measurements are 'incl' for light subjets and 'lt' for b and c subjets. "+\
"Scale factors are provided for the medium (M) and loose (L) working points."


def produce_subjet_json(df, tagger, year):
    subjet_json_sf = Correction.parse_obj(
        {
            "version": 1,
            "name": f"{tagger}_subjet",
            "description": f"{tagger} subjet tagging  factors for UL {year}. "+description,
            "inputs": [
                {
                    "name": "systematic", 
                    "type": "string"
                },
                {
                    "name": "method", 
                    "type": "string",
                    "description": "incl for light jets, lt for b/c jets"
                },
                {
                    "name": "working_point", 
                    "type": "string",
                    "description": "L/M"
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
            "generic_formulas": build_subjet_formulas(),
        }
    )
    return subjet_json_sf


