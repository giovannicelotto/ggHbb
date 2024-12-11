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

# ## Creating function to produce working point scale factors
def build_formula(sf):
    if len(sf) != 1:
        raise ValueError(sf)

    value = sf.iloc[0]["formula"]
    if "x" in str(value):
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
description = ""

description = "'up' and 'down' variations are available for the different measurement types. "+\
"The uncertainties are to be decorrelated between c jets ('wcharm'), b jets ('TnP') and light jets ('incl')"

def produce_wp_json(df, tagger, year):
    wp_json_sf = Correction.parse_obj(
        {
            "version": 2,
            "name": f"{tagger}_wp",
            "description": f"{tagger} fixedWP c-tagging factors for UL {year}. "+description,
            "inputs": [
                {
                    "name": "systematic", 
                    "type": "string"
                },
                {
                    "name": "method",
                    "type": "string",
                    "description": "incl for light jets, wcharm for b/c jets"
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
    )
    return wp_json_sf


