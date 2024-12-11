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

def build_formula(sf):
    if len(sf) != 1:
        raise ValueError(sf)

    value = sf.iloc[0]["formula"]
    return float(value)


def build_CvBbinning(sf):
    edges = sorted(set(sf["cvbMin"]) | set(sf["cvbMax"]))
    #print(f'build_ptbinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "CvB",
            "edges": edges,
            "content": [
                build_formula(sf[(sf["cvbMin"] >= lo) & (sf["cvbMax"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ],
            "flow": "clamp",
        }
    )

def build_CvLbinning(sf):
    edges = sorted(set(sf["cvlMin"]) | set(sf["cvlMax"]))
    #print(f'build_etabinning: {edges}')
    return Binning.parse_obj(
        {
            "nodetype": "binning",
            "input": "CvL",
            "edges": edges,
            "content": [
                build_CvBbinning(sf[(sf["cvlMin"] >= lo) & (sf["cvlMax"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ],
            "flow": "clamp",
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
                {"key": key, "value": build_CvLbinning(sf[sf["jetFlavor"] == key])}
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


description = "The scale factors have 13 default uncertainty sources "+\
"(Extrap, Interp, LHEScaleWeight_muF, LHEScaleWeight_muR, PSWeightFSR, PSWeightISR, PUWeight, Stat, XSec_BRUnc_DYJets_b, XSec_BRUnc_DYJets_c, XSec_BRUnc_WJets_c, jer, jesTotal). "+\
"All uncertainty sources are to be correlated across jet flavors. "+\
"All, except the 'Stat' uncertainty are to be correlated between years."

def produce_reshape_json(df, tagger, year):
    shape_json_sf = corr_deepCSV_UL18_shape = Correction.parse_obj(
        {
            "version": 1,
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
                    "name": "CvL",
                    "type": "real",
                    "description": f"{tagger} CvL value",
                },
                {
                    "name": "CvB",
                    "type": "real",
                    "description": f"{tagger} CvB value",
                },
            ],
            "output": {"name": "weight", "type": "real"},
            "data": build_systs(df.copy()),
        }
    )
    return shape_json_sf

