# %%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import re
import argparse
hep.style.use("CMS")
# %%
parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
parser.add_argument("-c", "--category", type=int, help="Index of the workspace")
parser.add_argument("-r", "--rebin", type=int, help="Rebinning")
parser.add_argument("-p", "--pdfIdx", type=int, help="Pdf Idx of Plotting")
parser.add_argument("-bl", "--blind_low", type=int, help="blind region")
parser.add_argument("-bh", "--blind_high", type=int, help="blind region")
args = parser.parse_args()
blind_low, blind_high = args.blind_low, args.blind_high
idx =args.category
path = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws%d.root"%args.category
f1 = ROOT.TFile.Open(path)
ws = f1.Get("ws3")
Signal_pdf_name = f"model_H_c{idx}"
Higgs_pdf = ws.pdf(Signal_pdf_name)


Background_pdf_name = f"CMS_hgg_0_2016_13TeV_bkgshape"
Background_pdf = ws.pdf(Background_pdf_name)
cat = ws.cat("pdfindex_%d_2016_13TeV"%args.category)
print("Number of categories:", cat.numTypes())

# Store all pdfs here
pdfs = []
it = cat.typeIterator()
t = it.Next()
while t:
    cat.setIndex(cat.lookupType(t.GetName()).getVal())
    print("Index:", cat.lookupType(t.GetName()).getVal(), "PDF : ", Background_pdf.getCurrentPdf().GetName())
    pdfs.append(Background_pdf.getCurrentPdf())
    t = it.Next()
pdf0 = Background_pdf.getCurrentPdf()
print("Available PDFs:", pdf0.GetName())


# Get Pdf Labels
pdfLabels = []
for pdf in pdfs:
    m = re.match(r"env_pdf_([A-Za-z]+)_(\d+)_cat", pdf.GetName())
    if m:
        method, num = m.groups()

        # mapping
        mapping = {
            "Bernstein": "Bern",
            "PowerLaw": "Pow",
            "Exponential": "Expo"
        }

        short = mapping.get(method, method)  # fallback to same if not mapped
        pdfLabel = f"{short}{num}"
        pdfLabels.append(pdfLabel)





# Normalizations
Background_norm_name = f"CMS_hgg_0_2016_13TeV_bkgshape_norm"
Background_pdf_norm = ws.var(Background_norm_name)
Higgs_norm_name = f"model_H_c{idx}_norm"
Higgs_pdf_norm = ws.var(Higgs_norm_name)

dijet_mass_var_name = f"dijet_mass_c{idx}"
dijet_mass_var = ws.var(dijet_mass_var_name)


print("Number of fitted Higgs : ", Higgs_pdf_norm.getVal())
print("Number of fitted Background : ", 103903)
# %%
assert isinstance(pdfs[args.pdfIdx], ROOT.RooAddPdf)

# Get list of sub-pdfs
subpdfs = pdfs[args.pdfIdx].pdfList()      # RooArgList
fraclist = pdfs[args.pdfIdx].coefList()    # RooArgList (fractions or yields depending on definition)

# Grab the first component
pdf0_comp0 = subpdfs.at(0)
pdf0_comp1 = subpdfs.at(1)

# If coefList are fractions, you can scale manually by your known norm
frac0 = fraclist.at(0).getVal()  # e.g. 0.7 if 70% fraction
norm_total = 103903
norm_comp0 = norm_total * frac0
norm_comp1 = norm_total * (1-frac0)


datahist = ws.data("rooHist_data_cat%d"%args.category)
Zhist = ws.data("rooHist_Z_cat%d"%args.category)
Hhist = ws.data("rooHist_H_cat%d"%args.category)

# Extract points (bin centers and contents)
rebin_factor = args.rebin
x_data = []
y_data = []
y_Z_hist = []
y_H_hist = []
y_err  = []
function_on_grid = []
binwidth = datahist.get(1).getRealValue(dijet_mass_var_name) - datahist.get(0).getRealValue(dijet_mass_var_name)
# Loop over bins
for i in range(datahist.numEntries()):
    point = datahist.get(i)   # returns a RooArgSet with variables
    x = point.getRealValue(dijet_mass_var_name)
    y = datahist.weight()     # bin content
    err = datahist.weightError(ROOT.RooAbsData.SumW2)  # stat error if stored
    dijet_mass_var.setVal(x)
    yb = pdf0_comp1.getVal(ROOT.RooArgSet(dijet_mass_var)) * norm_comp1*(binwidth)
    function_on_grid.append(yb)
    
    x_data.append(x)
    y_data.append(y)
    Zhist.get(i)
    Hhist.get(i)
    y_Z_hist.append(Zhist.weight())
    y_H_hist.append(Hhist.weight())
    y_err.append(err)
bins = list((np.array(x_data) - binwidth/2))
bins.append(x_data[-1]+binwidth)
x_data = np.array(x_data)
y_data = np.array(y_data)
y_err  = np.array(y_err)
function_on_grid = np.array(function_on_grid)
if rebin_factor>1:
    print("*"*30)
    print("\n"*20)
    function_on_grid = function_on_grid[:len(function_on_grid)//rebin_factor*rebin_factor].reshape(-1, rebin_factor).sum(axis=1)

# Range
xvals = np.linspace(50, 220, 500)  # 500 points between 50 and 220

# Prepare arrays
y_backgrounds = [[] for i in range(cat.numTypes())] # list of list
y_sig = []
y_comp0 = []
y_comp1 = []
for xv in xvals:
    dijet_mass_var.setVal(xv)
    
    # Evaluate pdf at this point (pdf gives density per unit of x)
    for i in range(cat.numTypes()):
        yb = pdfs[i].getVal(ROOT.RooArgSet(dijet_mass_var)) * 103903
        y_backgrounds[i].append(yb*(x_data[1]-x_data[0]))
    ys = Higgs_pdf.getVal(ROOT.RooArgSet(dijet_mass_var)) * Higgs_pdf_norm.getVal() * 10

    val0 = pdf0_comp0.getVal(ROOT.RooArgSet(dijet_mass_var)) * norm_comp0
    val1 = pdf0_comp1.getVal(ROOT.RooArgSet(dijet_mass_var)) * norm_comp1
    
    y_comp0.append(val0*(x_data[1]-x_data[0]))
    y_comp1.append(val1*(x_data[1]-x_data[0]))
    
    y_sig.append(ys*(x_data[1]-x_data[0]))

# Convert to numpy arrays
y_back = np.array(y_backgrounds)
y_sig = np.array(y_sig)

# Plot
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
ax[0].hist(bins[:-1], bins=bins, weights=np.array(y_Z_hist)*rebin_factor, label=r"$Z \rightarrow b\bar{b}$ (MC)", histtype='step', color='green', linewidth=2)
ax[1].hist(bins[:-1], bins=bins, weights=np.array(y_Z_hist)*rebin_factor, label=r"$Z \rightarrow b\bar{b}$ (MC)", histtype='step', color='green', linewidth=2)
color_cycle = ['orange', 'purple', 'brown', 'gray', 'darkgreen']
for i in range(cat.numTypes()):
    if i == args.pdfIdx:
        color='blue'
        label = f"Background ({pdfLabels[i]})"
    else:
        color = color_cycle[i % len(color_cycle)]
        label = f"Background ({pdfLabels[i]})"
    ax[0].plot(xvals, y_back[i,:]*rebin_factor, label=label, color=color)
#ax[0].plot(xvals, y_back[1,:]*rebin_factor, label="Background (%s)"%pdfLabels[1], color="purple")
#ax[0].plot(xvals, y_back[2,:]*rebin_factor, label="Background (%s)"%pdfLabels[2], color="orange")
ax[0].plot(xvals, np.array(y_comp1)*rebin_factor, label="Non-Resonant (%s)"%pdfLabels[args.pdfIdx],color="C0",  linestyle="--")
ax[0].plot(xvals, np.array(y_comp0)*rebin_factor, label=r"$Z \rightarrow b\bar{b}$ (%s)"%(pdfLabels[args.pdfIdx]), color="blue",  linestyle="--")
ax[0].plot(xvals, y_sig*rebin_factor, label=r"$H \rightarrow b\bar{b}$ x 10", color="red", linestyle="--")




if rebin_factor>1:
    nbins_new = len(y_data) // rebin_factor
    y_rebinned = y_data[:nbins_new*rebin_factor].reshape(-1, rebin_factor).sum(axis=1)
    yerr_rebinned = np.sqrt(y_rebinned)
    bins_rebinned = bins[::rebin_factor]
    if bins_rebinned[-1] != bins[-1]:
        bins_rebinned = np.append(bins_rebinned, bins[-1])
    x_rebinned = 0.5 * (np.array(bins_rebinned[1:]) + np.array(bins_rebinned[:-1]))
    mask = (x_rebinned < blind_high) | (x_rebinned > blind_high)

    ax[0].errorbar(
        x_rebinned[mask],
        y_rebinned[mask],
        yerr=yerr_rebinned[mask],
        fmt="o",
        color="black",
        label="Data"
    )
else:
    ax[0].errorbar(x_data[((x_data<blind_low) | (x_data>blind_high))], y_data[((x_data<blind_low) | (x_data>blind_high))], yerr=y_err[(x_data<blind_low) | (x_data>blind_high)], fmt="o", color="black", label="Data")


ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1]*1.3)
ax[0].text(x=0.025, y=0.9, s="Category Tight-Pass", transform=ax[0].transAxes, fontsize=18)
#ax[0].hist(bins[:-1], bins=bins, weights=np.array(y_H_hist)*10, label="MC H x 10")
ax[0].set_ylabel("Events / %.2f GeV"%((x_data[1]- x_data[0])*rebin_factor))
ax[0].legend()
ax[0].set_xlim(datahist.get(0).getRealValue(dijet_mass_var_name) - (x_data[1]- x_data[0])/2, datahist.get(datahist.numEntries()-1).getRealValue(dijet_mass_var_name) + (x_data[1]- x_data[0])/2)
if rebin_factor>1:
    ax[1].errorbar(x_rebinned[(x_rebinned<blind_low) | (x_rebinned>blind_high)], y_rebinned[(x_rebinned<blind_low) | (x_rebinned>blind_high)] - function_on_grid[(x_rebinned<blind_low) | (x_rebinned>blind_high)], yerr=yerr_rebinned[(x_rebinned<blind_low) | (x_rebinned>blind_high)], fmt="o", color="black", label="Data")
else:
    ax[1].errorbar(x_data[(x_data<blind_low) | (x_data>blind_high)], y_data[(x_data<blind_low) | (x_data>blind_high)] - function_on_grid[(x_data<blind_low) | (x_data>blind_high)], yerr=y_err[(x_data<blind_low) | (x_data>blind_high)], fmt="o", color="black", label="Data")
ax[1].plot(xvals, np.array(y_comp0)*rebin_factor, label=r"$Z \rightarrow b\bar{b}$", color="blue",  linestyle="--")
ax[1].plot(xvals, np.array(y_sig)*rebin_factor, label=r"$H \rightarrow b\bar{b}$ x 10", color="red", linestyle="--")
ax[1].hlines(y=0, xmin=datahist.get(0).getRealValue(dijet_mass_var_name) - (x_data[1]- x_data[0])/2,xmax= datahist.get(datahist.numEntries()-1).getRealValue(dijet_mass_var_name) + (x_data[1]- x_data[0])/2, color='black')
ax[1].set_xlabel(r"$m_{jj}$ [GeV]")
ax[1].set_ylabel("Residuals")
hep.cms.label(label="Private Work", data=True, lumi=41.6, ax=ax[0])
fig.savefig("/t3home/gcelotto/ggHbb/WSFit/scripts/plot_%d.png"%args.category, bbox_inches='tight')
