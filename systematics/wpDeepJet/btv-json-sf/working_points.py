from correctionlib import convert
from correctionlib.schemav2 import (
    Category,
    Correction,
)

btagging_wp = {}
btagging_wp["deepJet"] = {}
btagging_wp["deepJet"]["2018"] = {
    "L":    0.0490,
    "M":    0.2783,
    "T":    0.7100,
    }
btagging_wp["deepJet"]["2017"] = {
    "L":    0.0532,
    "M":    0.3040,
    "T":    0.7476,
    }
btagging_wp["deepJet"]["2016postVFP"] = {
    "L":    0.0480,
    "M":    0.2489,
    "T":    0.6377,
    }
btagging_wp["deepJet"]["2016preVFP"] = {
    "L":    0.0508,
    "M":    0.2598,
    "T":    0.6502,
    }
btagging_wp["deepCSV"] = {}
btagging_wp["deepCSV"]["2018"] = {
    "L":    0.1208,
    "M":    0.4168,
    "T":    0.7665,
    }
btagging_wp["deepCSV"]["2017"] = {
    "L":    0.1355,
    "M":    0.4506,
    "T":    0.7738,
    }
btagging_wp["deepCSV"]["2016postVFP"] = {
    "L":    0.1918,
    "M":    0.5847,
    "T":    0.8767,
    }
btagging_wp["deepCSV"]["2016preVFP"] = {
    "L":    0.2027,
    "M":    0.6001,
    "T":    0.8819,
    }

ctagging_wp = {}
ctagging_wp["deepJet"] = {}

jval8 = {"L": {}, "M": {}, "T": {}}
jval8["L"]["CvL"] = 0.038
jval8["M"]["CvL"] = 0.099
jval8["T"]["CvL"] = 0.282
jval8["L"]["CvB"] = 0.246
jval8["M"]["CvB"] = 0.325
jval8["T"]["CvB"] = 0.267
ctagging_wp["deepJet"]["2018"] = jval8

jval7 = {"L": {}, "M": {}, "T": {}}
jval7["L"]["CvL"] = 0.030
jval7["M"]["CvL"] = 0.085
jval7["T"]["CvL"] = 0.520
jval7["L"]["CvB"] = 0.400
jval7["M"]["CvB"] = 0.340
jval7["T"]["CvB"] = 0.050
ctagging_wp["deepJet"]["2017"] = jval7

jval61 = {"L": {}, "M": {}, "T": {}}
jval61["L"]["CvL"] = 0.039
jval61["M"]["CvL"] = 0.098
jval61["T"]["CvL"] = 0.270
jval61["L"]["CvB"] = 0.327
jval61["M"]["CvB"] = 0.370
jval61["T"]["CvB"] = 0.256
ctagging_wp["deepJet"]["2016preVFP"] = jval61

jval62 = {"L": {}, "M": {}, "T": {}}
jval62["L"]["CvL"] = 0.039
jval62["M"]["CvL"] = 0.099
jval62["T"]["CvL"] = 0.269
jval62["L"]["CvB"] = 0.305
jval62["M"]["CvB"] = 0.353
jval62["T"]["CvB"] = 0.247
ctagging_wp["deepJet"]["2016postVFP"] = jval62

ctagging_wp["deepCSV"] = {}

cval8 = {"L": {}, "M": {}, "T": {}}
cval8["L"]["CvL"] = 0.064
cval8["M"]["CvL"] = 0.153
cval8["T"]["CvL"] = 0.405
cval8["L"]["CvB"] = 0.313
cval8["M"]["CvB"] = 0.363
cval8["T"]["CvB"] = 0.288
ctagging_wp["deepCSV"]["2018"] = cval8

cval7 = {"L": {}, "M": {}, "T": {}}
cval7["L"]["CvL"] = 0.040
cval7["M"]["CvL"] = 0.144
cval7["T"]["CvL"] = 0.730
cval7["L"]["CvB"] = 0.345
cval7["M"]["CvB"] = 0.290
cval7["T"]["CvB"] = 0.100
ctagging_wp["deepCSV"]["2017"] = cval7

cval61 = {"L": {}, "M": {}, "T": {}}
cval61["L"]["CvL"] = 0.088
cval61["M"]["CvL"] = 0.181
cval61["T"]["CvL"] = 0.417
cval61["L"]["CvB"] = 0.214
cval61["M"]["CvB"] = 0.228
cval61["T"]["CvB"] = 0.138
ctagging_wp["deepCSV"]["2016preVFP"] = cval61

cval62 = {"L": {}, "M": {}, "T": {}}
cval62["L"]["CvL"] = 0.088
cval62["M"]["CvL"] = 0.180
cval62["T"]["CvL"] = 0.407
cval62["L"]["CvB"] = 0.204
cval62["M"]["CvB"] = 0.221
cval62["T"]["CvB"] = 0.136
ctagging_wp["deepCSV"]["2016postVFP"] = cval62

def produce_btagging_wp(tagger, year):
    json_sf = Correction.parse_obj(
        {
            "version": 1,
            "name": f"{tagger}_wp_values",
            "description": f"working points for {tagger} in UL {year}.",
            "inputs": [
                {
                "name": "working_point",
                "type": "string",
                "description": "L/M/T"
                },
            ],
            "output": {"name": "wp", "type": "real"},
            "data": Category.parse_obj(
                {
                    "nodetype": "category",
                    "input": "working_point",
                    "content": [
                        {"key": key, "value": btagging_wp[tagger][year][key]}
                        for key in btagging_wp[tagger][year]
                    ],
                }   
            )
        }
    )
    return json_sf

def produce_ctagging_wp(tagger, year):
    json_sf = Correction.parse_obj(
        {
            "version": 1,
            "name": f"{tagger}_wp_values",
            "description": f"working points for {tagger} in UL {year}. Important: the two axes 'CvB' and 'CvL' are to be used together, e.g. a loose working point is defined as a simultaneous cut on both values.",
            "inputs": [
                {
                "name": "working_point",
                "type": "string",
                "description": "L/M/T"
                },
                {
                "name": "axis",
                "type": "string",
                "description": "CvB/CvL cut"
                },
            ],
            "output": {"name": "wp", "type": "real"},
            "data": Category.parse_obj(
                {
                    "nodetype": "category",
                    "input": "working_point",
                    "content": [
                        {
                            "key": key, "value": {
                                "nodetype": "category",
                                "input": "axis",
                                "content": [
                                    {"key": axis, "value": ctagging_wp[tagger][year][key][axis]}
                                    for axis in ctagging_wp[tagger][year][key]
                                ],
                            }
                        }
                        for key in ctagging_wp[tagger][year]
                    ],
                }   
            )
        }
    )
    return json_sf
