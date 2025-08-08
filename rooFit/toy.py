#%%
import numpy as np
import pandas as pd
import ROOT

np.random.seed(123)

# Parametri per i 3 contributi
N_bkg = 100_000
N_Z = 5000
N_H = 100

# Fondo esponenziale decrescente
bkg_x = np.random.exponential(scale=100, size=N_bkg) + 50
bkg_x = bkg_x[bkg_x < 300]

# Picco Z gaussian centered @91 GeV
Z_x = np.random.normal(loc=91, scale=10, size=N_Z)
Z_x = Z_x[(Z_x > 50) & (Z_x < 300)]

# Picco Higgs gaussian centered @125 GeV
#H_x = np.random.normal(loc=125, scale=2, size=N_H)
#H_x = H_x[(H_x > 115) & (H_x < 135)]

# Combina tutto in DataFrame
#df = pd.DataFrame({"mass": np.concatenate([bkg_x, Z_x, H_x])})
df = pd.DataFrame({"mass": np.concatenate([bkg_x, Z_x])})
print(f"Tot eventi generati: {len(df)}")

# %%
import matplotlib.pyplot as plt
nbins=200
plt.hist(df.mass, bins=nbins)


# %%

# Crea TH1F da mass
bins = np.linspace(50, 300, nbins)  # 40 bin 50-180 GeV
hist = ROOT.TH1F("data_hist", "data", len(bins)-1, bins)




# Riempimento veloce con numpy histogram
counts, _ = np.histogram(df["mass"], bins=bins)
for i, c in enumerate(counts):
    hist.SetBinContent(i+1, c)
    hist.SetBinError(i+1, np.sqrt(c))  # Poisson errors


c1 = ROOT.TCanvas("c1", "c1", 800, 600)
c1.cd()
hist.SetLineColor(ROOT.kBlue + 1)
hist.SetLineWidth(2)
hist.Draw("E")
c1.SaveAs("/t3home/gcelotto/ggHbb/rooFit/c1.png")
# %%
# RooFit observable e RooDataHist
x = ROOT.RooRealVar("x", "mass", 50, 300)
data_roohist = ROOT.RooDataHist("data_obs", "Observed data", ROOT.RooArgList(x), hist)


# %%
# Fondo: pol2 * exp(-d*x)
a = ROOT.RooRealVar("a", "a", 1.0, -10, 10)
b = ROOT.RooRealVar("b", "b", 0.0, -0.1, )
c = ROOT.RooRealVar("c", "c", 0.0, -0.1, 0.1)
d = ROOT.RooRealVar("d", "d", 0.02, 0.0001, 1)

pdf_bkg = ROOT.RooGenericPdf("bkg", "bkg", "(a + b*x+c*x*x)^2*exp(-d*x)", ROOT.RooArgList(x, a, b, c, d))

# Z: gauss + cb con media condivisa
meanZ = ROOT.RooRealVar("meanZ", "meanZ", 91, 85, 95)

sigmaG = ROOT.RooRealVar("sigmaG", "sigmaG", 2, 0.5, 10)
gauss = ROOT.RooGaussian("gauss", "Gaussian", x, meanZ, sigmaG)

alpha = ROOT.RooRealVar("alpha", "alpha", 1.5, 0.5, 5)
n = ROOT.RooRealVar("n", "n", 2.0, 0.5, 10)
sigmaCB = ROOT.RooRealVar("sigmaCB", "sigmaCB", 2, 0.5, 10)
cb = ROOT.RooCBShape("cb", "Crystal Ball", x, meanZ, sigmaCB, alpha, n)

fracG = ROOT.RooRealVar("fracG", "fracG", 0.5, 0.45, 0.55)
pdf_Ztemp = ROOT.RooAddPdf("pdf_Ztemp", "gauss + cb", ROOT.RooArgList(gauss, cb), ROOT.RooArgList(fracG))

# Higgs: simile con media attorno 125 GeV
#meanH = ROOT.RooRealVar("meanH", "meanH", 125, 120, 130)
#sigmaG_H = ROOT.RooRealVar("sigmaG_H", "sigmaG_H", 2, 0.1, 5)
#gauss_H = ROOT.RooGaussian("gauss_H", "Gaussian_H", x, meanH, sigmaG_H)
#
#alpha_H = ROOT.RooRealVar("alpha_H", "alpha_H", 1.5, 0.5, 5)
#n_H = ROOT.RooRealVar("n_H", "n_H", 2.0, 0.5, 10)
#sigmaCB_H = ROOT.RooRealVar("sigmaCB_H", "sigmaCB_H", 2, 0.1, 5)
#cb_H = ROOT.RooCBShape("cb_H", "Crystal Ball H", x, meanH, sigmaCB_H, alpha_H, n_H)
#
#fracG_H = ROOT.RooRealVar("fracG_H", "fracG_H", 0.5, 0, 1)
#pdf_Htemp = ROOT.RooAddPdf("pdf_Htemp", "gauss_H + cb_H", ROOT.RooArgList(gauss_H, cb_H), ROOT.RooArgList(fracG_H))

# %%
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", len(bkg_x), 0, 1e6)
n_Z = ROOT.RooRealVar("n_Z", "n_Z", len(Z_x), 0, 1e5)
#n_H = ROOT.RooRealVar("n_H", "n_H", len(H_x), 0, 1e5)

model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(pdf_bkg, pdf_Ztemp),
                       ROOT.RooArgList(n_bkg, n_Z))

x.setRange("R1", 50,    100)
x.setRange("R2", 150,   300)

fit_result = model.fitTo(
    data_roohist,
    Range= "R1,R2",
    Save=True

)
#%%
fit_result.Print()                   # Print fit summary to console
fit_result.covQual()                # Covariance matrix quality (3 = good)
fit_result.status()                # MINUIT status (0 = success)
fit_result.edm()  

# %%
frame = x.frame()
data_roohist.plotOn(frame)
model.plotOn(frame, ROOT.RooFit.Components("bkg"), ROOT.RooFit.LineStyle(ROOT.kDashed))
chi2_ndof = frame.chiSquare()
model.plotOn(frame, ROOT.RooFit.Components("pdf_Ztemp"), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed))
model.plotOn(frame)  # total fit

# Optional: Plot components
#model.plotOn(frame, ROOT.RooFit.Components("pdf_Htemp"), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kGreen+2))

c1 = ROOT.TCanvas("c1", "Fit", 800, 600)
frame.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/rooFit/fit_model.png")
# %%
# Create pull histogram (Data - Fit)
pull_hist = frame.pullHist()

# Frame for the pull (residuals)
pull_frame = x.frame()
pull_frame.addPlotable(pull_hist, "P")
pull_frame.SetTitle("Data - Fit Residuals")
pull_frame.GetYaxis().SetTitle("Pull")
pull_frame.GetYaxis().SetTitleOffset(1.2)
pull_frame.GetYaxis().SetNdivisions(505)
pull_frame.GetYaxis().SetLabelSize(0.1)
pull_frame.GetXaxis().SetLabelSize(0.1)
pull_frame.GetXaxis().SetTitleSize(0.12)
pull_frame.GetYaxis().SetTitleSize(0.12)

# Create canvas and pads
canvas = ROOT.TCanvas("canvas", "Fit with Residuals", 800, 800)
pad1 = ROOT.TPad("pad1", "Top pad", 0, 0.3, 1, 1.0)
pad2 = ROOT.TPad("pad2", "Bottom pad", 0, 0.05, 1, 0.3)
pad1.SetBottomMargin(0.02)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.3)
pad1.Draw()
pad2.Draw()

# Draw main plot
pad1.cd()
frame.Draw()

print(f"Chi2/ndof: {chi2_ndof}")
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.DrawLatex(0.65, 0.85, f"#chi^{{2}}/ndf = {chi2_ndof:.2f}")

#blinded_box = ROOT.TBox(120, 0, 130, frame.GetMaximum())
#blinded_box.SetFillColor(ROOT.kGray+2)
#blinded_box.SetFillStyle(3002)  # semi-transparent hatch
#blinded_box.Draw("same")

# Draw residuals
pad2.cd()
pull_frame.Draw()

# Save canvas
canvas.SaveAs("/t3home/gcelotto/ggHbb/rooFit/fit_model_with_residuals.png")



# %%
w = ROOT.RooWorkspace("w")
getattr(w, "import")(x)
getattr(w, "import")(data_roohist)
getattr(w, "import")(model)

w.writeToFile("workspace.root")

# %%
