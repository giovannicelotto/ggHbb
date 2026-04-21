#def apply_systematic(df, syst_name, direction):
#    if syst_name is None:
#        return df
#
#    df = df.copy()
#
#    if syst_name == "JES":
#        # example: shifted variable
#        shift = df[f"PNN_qm_{direction}"]
#        df["PNN_qm"] = shift
#
#    elif syst_name == "PU":
#        df["weight"] *= df[f"puWeight{direction}"]
#
#    return df