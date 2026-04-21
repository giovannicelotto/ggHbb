def prepare_mc(dfMC, lumi):
    dfMC = dfMC.copy()

    # reweight
    dfMC["weight"] *= lumi
    return dfMC


def apply_cuts(dfData, dfMC, config):
    cuts = config["cuts"]

    return (
        dfData.query(cuts),
        dfMC.query(cuts)
    )

def apply_cuts_single(df, config):
    '''
    Apply cuts on single dataframe
    '''
    cuts = config["cuts"]
    return df.query(cuts)