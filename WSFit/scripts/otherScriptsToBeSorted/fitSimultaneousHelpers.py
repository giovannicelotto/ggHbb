import ROOT
def draw_parameters_simple(params, x=0.15, y=0.7, dy=0.04, textSize=0.03):
    """
    Draw parameter name, value, error on current canvas using TLatex.
    """
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextAlign(31)
    txt.SetTextSize(textSize)
    ypos = y
    for p in params:
        # Only draw if it is a RooRealVar
        if isinstance(p, ROOT.RooRealVar):
            if "model_Z" in p.GetName():
                pass
            elif p.getError() < 1e-7:
                continue

            line = f"{p.GetName()} = {p.getVal():.3f} +- {p.getError():.5f}"
            txt.DrawLatex(x, ypos, line)
            ypos -= dy
    return txt