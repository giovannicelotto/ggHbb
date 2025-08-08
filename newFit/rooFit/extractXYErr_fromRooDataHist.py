import numpy as np
import ROOT

def extract_xy_yerr_from_roodatahist(datahist: ROOT.RooDataHist, varname: str):
    """Extract x, y, yerr from a RooDataHist.
    
    Parameters:
        datahist (ROOT.RooDataHist): The RooDataHist object.
        varname (str): The name of the variable (e.g., 'x').

    Returns:
        x (np.ndarray): Bin centers.
        y (np.ndarray): Bin contents (weights).
        yerr (np.ndarray): Bin errors.
    """
    x_vals = []
    y_vals = []
    y_errs = []

    n_bins = datahist.numEntries()
    var = datahist.get().find(varname)
    if not isinstance(var, ROOT.RooRealVar):
        raise TypeError(f"{varname} is not a RooRealVar")

    binning = var.getBinning()
    n_bins = binning.numBins()
    edges = np.array([binning.binLow(i) for i in range(n_bins)] + [binning.binHigh(n_bins - 1)])

    for i in range(n_bins):
        point = datahist.get(i)
        val = point.getRealValue(varname)
        weight = datahist.weight()
        error = datahist.weightError()

        x_vals.append(val)
        y_vals.append(weight)
        y_errs.append(error)

    return np.array(edges), np.array(y_vals), np.array(y_errs)
