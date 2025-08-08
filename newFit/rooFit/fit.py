# %%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplhep as hep
from scipy.stats import chi2
hep.style.use("CMS")
import sys
from extractXYErr_fromRooDataHist import *
ROOT.gSystem.CompileMacro("/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc", "kf")
parser = argparse.ArgumentParser( prog='fit', description='Fit more categories simultaneously', epilog='Text at the bottom of help')
parser.add_argument('-c', '--cat', type=int, default=2)
parser.add_argument('-f', '--function', type=int, default=16)
parser.add_argument('-fit', '--fit', type=int, default=1)
parser.add_argument('-chi2', '--chi2', type=int, default=0)




if __name__ == '__main__' and not hasattr(sys, 'ps1'):
    args = parser.parse_args()
else:
    # In interactive mode
    args = parser.parse_args([]) 


# %%

# Open the file


f = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_sig.root")
w = f.Get("workspace_sig")
totalBackground_pdf  = w.pdf("model_f%d_cat%d"%(args.function, args.cat))
qcd_total_pdf        = w.pdf("f%d_c%d"%(args.function, args.cat))
Z_pdf                = w.pdf("model_Z_c%d"%args.cat)
data_hist            = w.data("rooHist_data_cat%d"%args.cat)
Z_hist               = w.data("rooHist_Z_cat%d"%args.cat)
data_hist_broad            = w.data("rooHist_data_broad_cat%d"%args.cat)
Z_hist_broad               = w.data("rooHist_Z_broad_cat%d"%args.cat)
x                    = w.var("dijet_mass_c%d"%args.cat) 

# Extract parameters for norm
nZ = w.var("nZ_cat%d"%args.cat)
mu = w.var("mu")
nQCD = w.var("nQCD_cat%d"%args.cat)


# Select the ranges
if args.cat==1:
    t0,t1,t2,t3 = 50, 139.99, 140, 300
elif args.cat==2:
    t0,t1,t2,t3 = 53, 140, 140, 200

x.setRange("R1", t0,   t1)
x.setRange("R2", t2,   t3)
nbins = data_hist.numEntries()
x1, x2 = x.getMin(), x.getMax()
# %%


# Perform the fit in the range
if args.fit:
    if args.chi2:
        fit_result = totalBackground_pdf.chi2FitTo(data_hist,
                      ROOT.RooFit.SumW2Error(True), 
                       ROOT.RooFit.Save(),
                       Range="R1,R2",)
    else:

        fit_result = totalBackground_pdf.fitTo(data_hist,
                                           #ROOT.RooFit.IntegrateBins(1e-2),
                                           ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(0), 
                           ROOT.RooFit.Strategy(2),
                           ROOT.RooFit.Extended(True), # To save normalization
                           ROOT.RooFit.SumW2Error(True), 
                           ROOT.RooFit.Save(),
                           Range="R1,R2",
                           )

    params = fit_result.floatParsFinal()  # Get the list of final floating parameters
    for i in range(params.getSize()):
        p = params.at(i)
        print(f"{p.GetName()} = {p.getVal():.7f} ± {p.getError():.4f}")

    n_pars = params.getSize()
    print("n_pars is ",n_pars)
else:
    n_pars = 8 # reasonable number just to plo












# %%
#Extract x and y
binwidth = data_hist.get(1).getRealValue("dijet_mass_c%d"%args.cat) - data_hist.get(0).getRealValue("dijet_mass_c%d"%args.cat)
x_vals = []
y_data = []

y_pdf_vals = []
y_qcd_vals = []
y_Z_vals = []


for i in range(nbins):
    x_ = data_hist.get(i).getRealValue("dijet_mass_c%d"%args.cat)  # or your var name
    x_vals.append(x_)

    y = data_hist.weight(i)                  # bin content (counts)
    y_data.append(y)
    
    x.setVal(x_)
    y_pdf = totalBackground_pdf.getVal()
    y_qcd = qcd_total_pdf.getVal()
    y_Z = Z_pdf.getVal()
    y_pdf_vals.append(y_pdf)    
    y_qcd_vals.append(y_qcd)    
    y_Z_vals.append(y_Z)    

y_pdf_vals, x_vals, y_data = np.array(y_pdf_vals), np.array(x_vals), np.array(y_data)
y_err = np.sqrt(y_data)
y_qcd_vals = np.array(y_qcd_vals)
y_Z_vals = np.array(y_Z_vals)

mChi2 = ((x_vals>t0) & (x_vals<t1)| (x_vals>t2) & (x_vals<t3))

#y_pdf_extended = y_qcd_vals/np.sum(y_qcd_vals[mChi2])*(nQCD.getVal()) + y_Z_vals*binwidth/np.sum(y_Z_vals*binwidth)*(nZ.getVal()*mu.getVal())

# Normalize the Z to all the region
y_Z_extended = y_Z_vals/np.sum(y_Z_vals)          *(nZ.getVal()*mu.getVal())
y_qcd_extended = y_qcd_vals/np.sum(y_qcd_vals)*(nQCD.getVal())
y_pdf_extended = y_Z_extended + y_qcd_extended


print("Extended yields:")
print("Z   : ", np.sum(y_Z_extended))
print("QCD : ", np.sum(y_qcd_extended))
print("All : ", np.sum(y_pdf_extended))





# CHI2
ndof = np.sum(mChi2) - n_pars
maskUnblind = (x_vals<t1) | (x_vals>t2)
chi2_stat = np.sum(((y_data[mChi2] - y_pdf_extended[mChi2])/y_err[mChi2])**2)
print(f"{chi2_stat:.1f}/{ndof}")

# %%
#x_vals = []
#y_data = []
#
#y_pdf_vals = []
#y_qcd_vals = []
#y_Z_vals = []
#
#Z_hist_broad
#
#nNewBins = data_hist_broad.numEntries()
#for i in range(nNewBins):
#    x_ = data_hist_broad.get(i).getRealValue("dijet_mass_c%d"%args.cat)  # or your var name
#    x_vals.append(x_)
#
#    y = data_hist_broad.weight(i)                  # bin content (counts)
#    y_data.append(y)
#    
#    x.setVal(x_)
#    y_pdf = totalBackground_pdf.getVal()
#    y_qcd = qcd_total_pdf.getVal()
#    y_Z = Z_pdf.getVal()
#    y_pdf_vals.append(y_pdf)    
#    y_qcd_vals.append(y_qcd)    
#    y_Z_vals.append(y_Z)    
## %%
#y_pdf_vals, x_vals, y_data = np.array(y_pdf_vals), np.array(x_vals), np.array(y_data)
#y_err = np.sqrt(y_data)
#y_qcd_vals = np.array(y_qcd_vals)
#y_Z_vals = np.array(y_Z_vals)
#y_Z_extended = y_Z_vals/np.sum(y_Z_vals)          *(nZ.getVal()*mu.getVal())
#y_qcd_extended = y_qcd_vals/np.sum(y_qcd_vals)*(nQCD.getVal())
#y_pdf_extended = y_Z_extended + y_qcd_extended
## CHI2
#print("Here starts chi2")
#mChi2 = ((x_vals>t0) & (x_vals<t1)| (x_vals>t2) & (x_vals<t3))
#print(np.sum(mChi2))
#print(n_pars)
#ndof = np.sum(mChi2) - n_pars
#maskUnblind = (x_vals<t1) | (x_vals>t2)
#chi2_stat = np.sum(((y_data[mChi2] - y_pdf_extended[mChi2])/y_err[mChi2])**2)
#print(f"{chi2_stat:.1f}/{ndof}")

print("Number of Z boson for Normalization : %d "%(nZ.getVal()))
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x_vals[maskUnblind], y_data[maskUnblind], yerr=y_err[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
ax[0].set_ylim(0, ax[0].get_ylim()[1])
#ax[0].set_xlim(50, 220)
bins, y_, yerr_ = extract_xy_yerr_from_roodatahist(Z_hist, varname='dijet_mass_c%d'%args.cat)
ax[1].hist(bins[:-1], bins=bins, weights=y_, label='Z')
ax[0].plot(x_vals, y_pdf_extended, label="S+B Fit", color='red')
ax[0].plot(x_vals, y_qcd_extended+y_Z_extended, label="Check", color='purple', linestyle='dashed')
ax[0].plot(x_vals, y_qcd_extended, label="B only", color='purple', linestyle='dashed')
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
print("pval %.2f"%chi2_pvalue)
ax[0].text(x=0.95, y=0.75, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)


ax[0].fill_between(x_vals, 0, max(y_data)*1.2, where=mChi2, color='green', alpha=0.2, label='Fit Region')
ax[1].set_ylabel("Counts per %.2f GeV"%(x_vals[1]-x_vals[0]))
#ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
ax[1].errorbar(x_vals[mChi2], (y_data-y_qcd_extended)[mChi2], yerr=y_err[mChi2], fmt='o', color='black', markersize=3)
ax[1].plot(x_vals, y_Z_extended, color='red')
#ax[1].set_ylim(-300,1500)
#ax[1].set_ylim(-5, 5)
ax[1].fill_between(x_vals, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=mChi2, color='green', alpha=0.2)
#ax[1].hlines(y=0, xmin=ax[0].get_xlim()[0], xmax=ax[0].get_xlim()[1], color='red')
ax[0].tick_params(labelsize=24)
ax[1].tick_params(labelsize=24)
ax[1].set_ylabel("Pulls")
ax[0].legend(fontsize=24)
#ax.set_ylabel("Data-PDF")
outName = "/t3home/gcelotto/ggHbb/newFit/rooFit/output/residuals_%d_f%d.png"%(args.cat, args.function)
print("Saved %s"%outName)
fig.savefig(outName)







# %%
# Plot obtained from frame
frame = x.frame(ROOT.RooFit.Title("Fit to data"))
data_hist.plotOn(frame, ROOT.RooFit.Name(f"rooHist_data_cat{args.cat}"),
            ROOT.RooFit.MarkerSize(0.2))
x.setRange("fullRange", 53, 200)
totalBackground_pdf.plotOn(frame, ROOT.RooFit.Range("fullRange"),
                           ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name(f"model_SB_c{args.cat}"), ROOT.RooFit.LineWidth(2))




c = ROOT.TCanvas("c", "Fit Canvas", 800, 600)
c.SetLeftMargin(0.15)
c.SetRightMargin(0.05)
c.SetTopMargin(0.1)
c.SetBottomMargin(0.15)
frame.Draw()
curve = frame.getObject(1)  # model (index may vary)
hist = frame.getObject(0)   # data

# Debug: loop through bins
for i in range(hist.GetN()):
    x_bin = hist.GetX()[i]
#    if (x_bin >105) & (x_bin<140):
#        continue
    y_data_bin = hist.GetY()[i]
    y_model_bin = curve.Eval(hist.GetX()[i])
    y_err_bin = hist.GetErrorY(i)

    chi2_bin = ((y_data_bin - y_model_bin) / y_err_bin)**2
    #print(f"Bin {i}: x = {x_bin:.2f}, Data = {y_data_bin:.2f}, Model = {y_model_bin:.2f}, MyModel = {y_pdf_extended[i]:.2f} Ratio = {(y_pdf_extended[i]-y_model_bin)/y_model_bin}Err = {y_err_bin:.2f}, Chi² = {chi2_bin:.2f}")


# Calculate chi2
chi2_sum = 0
n_points = 0
for i in range(hist.GetN()):
    x_bin = hist.GetX()[i]
    if not ((t0 <= x_bin <= t1) or (t2 <= x_bin <= t3)):
        continue  # skip points in the blinded region

    y_data_bin = hist.GetY()[i]
    y_model_bin = curve.Eval(x_bin)
    y_err_bin = hist.GetErrorY(i)
    
    chi2_bin = ((y_data_bin - y_model_bin) / y_err_bin) ** 2
    chi2_sum += chi2_bin
    n_points += 1

    #print(f"Bin {i}: x = {x_bin:.2f}, Data = {y_data_bin:.2f}, Model = {y_model_bin:.2f}, Err = {y_err_bin:.2f}, Chi² = {chi2_bin:.2f}")
print(f"Chi2 sum in sidebands = {chi2_sum:.2f} over {n_points} points")
print(f"Reduced Chi2 = {chi2_sum / (n_points - n_pars):.2f}")  # subtract n_pars parameters for degrees of freedom


#frame.GetXaxis().SetRangeUser(66,70)
#frame.GetYaxis().SetRangeUser(22e3,23e3)
# Add text box to canvas
latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.03)
latex.DrawLatex(0.6, 0.85, f"#chi^{{2}}/ndof = {chi2_stat/ndof:.2f} ({chi2_stat:.1f}/{ndof})")
print(f"Chi2 = {chi2_stat:.2f}")
print(f"NDOF = {ndof}")
print(f"Chi2 / NDOF = {chi2_stat/ndof:.3f}")
c.SaveAs(f"/t3home/gcelotto/ggHbb/newFit/rooFit/output/fit_result_{args.cat}_f{args.function}.pdf")
# %%


