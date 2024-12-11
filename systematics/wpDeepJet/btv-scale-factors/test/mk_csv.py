import pandas as pd

class CSVFile:
    def __init__(self, tagger, descriptor, version):
        self._name = f"{tagger}_{descriptor}_v{version}.csv"
        self._data = {
            "wp": [],
            "type": [],
            "syst": [],
            "flav": [],
            "etaMin": [],     
            "etaMax": [],
            "ptMin": [],
            "ptMax": [],                
            "discrMin": [],             
            "discrMax": [],
            "formula": []
        }

        self._types = ["shape",
            "comb", "mujets", "light", 
            "ptrel", "sys8", "ltsv", "tnp", "kinfit", 
            "wc"
            ]
        self._wps = ["-", 
            "L", "M", "T", "XT", "XXT"
            ]
        self._flavs = [0, 4, 5]
        
    def add_line(self, type, formula, 
            wp="-", syst="central", flav=5, 
            etaMin=0, etaMax=2.5, 
            ptMin=20, ptMax=1000, 
            discrMin=0, discrMax=1):

        # verify content
        if not type in self._types:
            raise ValueError(f"Invalid value type={type}")
        if not wp in self._wps:
            raise ValueError(f"Invalid value wp={wp}")
        if not int(flav) in self._flavs:
            raise ValueError(f"Invalid value flav={flav}")
        if not syst=="central" or syst.startswith("up") or syst.startswith("down"):
            raise ValueError(f"Invalid value syst={syst}")
        
        if float(etaMin) < 0:
            raise ValueError(f"Invalid value etaMin={etaMin}")
        if float(etaMin) >= float(etaMax):
            raise ValueError(f"Invalid value etaMin={etaMin} >= etaMax={etaMax}")
        if float(ptMin) >= float(ptMax):
            raise ValueError(f"Invalid value ptMin={ptMin} >= ptMax={ptMax}")
        if float(discrMin) >= float(discrMax):
            raise ValueError(f"Invalid value discrMin={etaMin} >= discrMax={discrMax}")
        
        self._data["wp"].append(str(wp))
        self._data["type"].append(str(type))
        self._data["syst"].append(str(syst))
        self._data["flav"].append(int(flav))
        self._data["etaMin"].append(float(etaMin))
        self._data["etaMax"].append(float(etaMax))
        self._data["ptMin"].append(int(ptMin))
        self._data["ptMax"].append(int(ptMax))
        self._data["discrMin"].append(float(discrMin))
        self._data["discrMax"].append(float(discrMax))
        self._data["formula"].append(str(formula))

    def to_df(self):
        import pandas as pd
        df = pd.DataFrame.from_dict(self._data) 
        return df

    def write(self, path="."):
        import os
        out_file = os.path.join(path, self._name)
        df = self.to_df()
        df.to_csv(out_file, index=False)
        print(f"Wrote csv file to {out_file}")

    def __str__(self):
        import pandas as pd
        print(self.to_df())
        return ""

