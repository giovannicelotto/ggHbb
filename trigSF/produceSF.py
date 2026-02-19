# %%
# Nota Questo Script é sbagliato!
# Nota Questo Script é sbagliato!
# Nota Questo Script é sbagliato!
# Nota Questo Script é sbagliato!
# Lo SF corretto é (1- (1-edata1)*(1-edata2))  /  (1- (1-eMC1)*(1-eMC2))
import ROOT
import sys

ROOT.gROOT.SetBatch(True)

# Input files
f_data = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/trigSF/trgMu_effData_UL.root", "READ")
f_mc   = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/trigSF/trgMu_effMC_UL.root", "READ")
# %%
if not f_data or not f_mc:
    raise RuntimeError("Error opening input files")

h_data = f_data.Get("hMap")
h_mc   = f_mc.Get("hMap")

if not h_data or not h_mc:
    raise RuntimeError("Histogram hMap not found in one of the files")

# Clone histograms to preserve binning and axes
h_ratio = h_data.Clone("effData_over_effMC")
h_ratio.Reset("ICES")

h_comp = h_data.Clone("oneMinusEffData_over_oneMinusEffMC")
h_comp.Reset("ICES")

# Loop over bins
for ix in range(1, h_data.GetNbinsX() + 1):
    for iy in range(1, h_data.GetNbinsY() + 1):

        eff_d = h_data.GetBinContent(ix, iy)
        eff_m = h_mc.GetBinContent(ix, iy)

        # eff_Data / eff_MC
        if eff_m > 0:
            h_ratio.SetBinContent(ix, iy, eff_d / eff_m)
        else:
            h_ratio.SetBinContent(ix, iy, 0.0)

        # (1-eff_Data)/(1-eff_MC)
        denom = 1.0 - eff_m
        if denom > 0:
            h_comp.SetBinContent(ix, iy, (1.0 - eff_d) / denom)
        else:
            h_comp.SetBinContent(ix, iy, 0.0)

# Output file (same structure style as input)
f_out = ROOT.TFile.Open("eff_ratios.root", "RECREATE")
h_ratio.Write()
h_comp.Write()
f_out.Close()

f_data.Close()
f_mc.Close()

# %%


print("Plotting Eff data / Eff MC")


ROOT.gStyle.SetOptStat(0)
f = ROOT.TFile.Open("eff_ratios.root")
h_orig = f.Get("effData_over_effMC")

nx = h_orig.GetNbinsX()
ny = h_orig.GetNbinsY()

# Uniform display histogram
h_disp = ROOT.TH2D(
    "h_disp",
    "Scale factors; Muon p_{T}; d_{xy}/#sigma(d_{xy})",
    nx, 0, nx,
    ny, 0, ny
)

# Copy contents by bin index
for ix in range(1, nx + 1):
    for iy in range(1, ny + 1):
        h_disp.SetBinContent(ix, iy, h_orig.GetBinContent(ix, iy))
        h_disp.SetBinError(ix, iy, h_orig.GetBinError(ix, iy))

# Build correct labels from original bin edges
xaxis_orig = h_orig.GetXaxis()
yaxis_orig = h_orig.GetYaxis()

xaxis_disp = h_disp.GetXaxis()
yaxis_disp = h_disp.GetYaxis()

for ix in range(1, nx + 1):
    lo = xaxis_orig.GetBinLowEdge(ix)
    hi = xaxis_orig.GetBinUpEdge(ix)
    xaxis_disp.SetBinLabel(ix, f"{lo:g}-{hi:g}")

for iy in range(1, ny + 1):
    lo = yaxis_orig.GetBinLowEdge(iy)
    hi = yaxis_orig.GetBinUpEdge(iy)
    yaxis_disp.SetBinLabel(iy, f"{lo:g}-{hi:g}")

# Style
xaxis_disp.LabelsOption("h")
xaxis_disp.SetTickLength(0)
yaxis_disp.SetTickLength(0)

# Draw
c = ROOT.TCanvas("c", "", 900, 800)
c.SetLeftMargin(0.12)
h_disp.Draw("COLZ TEXT")
c.SaveAs("SF_uniformDisplay.png")

# %%





print("Plotting 1-EffData over 1-EffMC")
ROOT.gStyle.SetOptStat(0)
f = ROOT.TFile.Open("eff_ratios.root")
h_comp = f.Get("oneMinusEffData_over_oneMinusEffMC;1")

nx = h_comp.GetNbinsX()
ny = h_comp.GetNbinsY()

# Uniform display histogram
h_disp = ROOT.TH2D(
    "h_disp",
    "Scale factors; Muon p_{T}; d_{xy}/#sigma(d_{xy})",
    nx, 0, nx,
    ny, 0, ny
)

# Copy contents by bin index
for ix in range(1, nx + 1):
    for iy in range(1, ny + 1):
        h_disp.SetBinContent(ix, iy, h_comp.GetBinContent(ix, iy))
        h_disp.SetBinError(ix, iy, h_comp.GetBinError(ix, iy))

# Build correct labels from original bin edges
xaxis_orig = h_comp.GetXaxis()
yaxis_orig = h_comp.GetYaxis()

xaxis_disp = h_disp.GetXaxis()
yaxis_disp = h_disp.GetYaxis()

for ix in range(1, nx + 1):
    lo = xaxis_orig.GetBinLowEdge(ix)
    hi = xaxis_orig.GetBinUpEdge(ix)
    xaxis_disp.SetBinLabel(ix, f"{lo:g}-{hi:g}")

for iy in range(1, ny + 1):
    lo = yaxis_orig.GetBinLowEdge(iy)
    hi = yaxis_orig.GetBinUpEdge(iy)
    yaxis_disp.SetBinLabel(iy, f"{lo:g}-{hi:g}")

# Style
xaxis_disp.LabelsOption("h")
xaxis_disp.SetTickLength(0)
yaxis_disp.SetTickLength(0)

# Draw
c = ROOT.TCanvas("c", "", 900, 800)
c.SetLeftMargin(0.12)
h_disp.Draw("COLZ TEXT")
c.SaveAs("SF_notTrig_uniformDisplay.png")