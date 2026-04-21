# %%
import numpy as np
import matplotlib.pyplot as plt
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError 
import mplhep as hep
from scipy.integrate import simpson

hep.style.use("CMS")

# %%
path_workspaces = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/"
category = 8
rebin_factor = 10

# %%
file = ROOT.TFile.Open(path_workspaces + f"ws{category}.root")
ws3 = file.Get("ws3")

cat = ws3.cat(f"pdfindex_{category}_2016_13TeV")

#get the default value of cat
bestPdfIndex = cat.getIndex()
print("***"*20)
print("best pdf index is ", bestPdfIndex)
print("***"*20)


data_hist = ws3.data(f"rooHist_data_cat{category}")
var_name = f"dijet_mass_c{category}"
mass = ws3.var(var_name)

# %%
def get_hist_data(data_hist):
    x_data, y_data = [], []

    for i in range(data_hist.numEntries()):
        point = data_hist.get(i)
        x_val = point.getRealValue(var_name)
        y_val = data_hist.weight()

        x_data.append(x_val)
        y_data.append(y_val)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_err = np.sqrt(y_data)

    return x_data, y_data, y_err


def rebin(x, y, yerr, factor):
    n = len(x) // factor * factor
    x = x[:n].reshape(-1, factor).mean(axis=1)
    y = y[:n].reshape(-1, factor).sum(axis=1)
    yerr = np.sqrt((yerr[:n].reshape(-1, factor)**2).sum(axis=1))
    return x, y, yerr


def normalize(y, x, data, category):
    integral = simpson(y, x)

    point =data_hist.get(1)
    x1 = point.getRealValue("dijet_mass_c%d"%category)
    point =data_hist.get(0)
    x0 = point.getRealValue("dijet_mass_c%d"%category)
    bw = x1-x0
    print(f"Binwidth from data histogram: {bw}")

    return y * (data_hist.sumEntries() * bw / integral)


# %%
# Data
x_data, y_data, y_err = get_hist_data(data_hist)
x_data, y_data, y_err = rebin(x_data, y_data, y_err, rebin_factor)

# %%
# x grid
x = np.linspace(50, 300, 1000)

# number of pdfs in multipdf
n_pdf = cat.numTypes()
import yaml

yaml_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{category}_pdfnames.yaml"

with open(yaml_path, "r") as f:
    pdf_dict = yaml.safe_load(f)

pdf_names = pdf_dict["pdf_names"]
# %%
# Loop over pdfindex and plot all curves, highlighting best PDF and Data - nonRes

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3, 1]}, sharex=True, figsize=(10, 8))
fig.align_labels()
# Plot rebinned data
ax[0].errorbar(x_data[(x_data<110) | (x_data>140) ], y_data[(x_data<110) | (x_data>140) ], yerr=y_err[(x_data<110) | (x_data>140) ], fmt='o', color='black', label="Data")
ax[0].set_xlim(x_data[0]-10, x_data[-1]+10)

# Loop over PDFs in the MultiPdf
n_pdf = cat.numTypes()
for i in range(n_pdf):
    #if i ==1:
    #    continue
    # Activate PDF i
    cat.setIndex(i)

    # --- Evaluate PDF on fine grid x ---
    y_bkg = []
    y_Z   = []
    y_H = []
    for val in x:
        mass.setVal(val)
        y_bkg.append(ws3.pdf("CMS_hgg_0_2016_13TeV_bkg_noZ").getVal())
        y_Z.append(ws3.pdf(f"model_Z_c{category}").getVal())
        y_H.append(ws3.pdf(f"model_H_c{category}").getVal())
    y_bkg = np.array(y_bkg)
    y_Z   = np.array(y_Z)
    y_H   = np.array(y_H)

    print("Simpson here : ", simpson(y_Z, x))
    print("Simpson here : ", simpson(y_H, x)) # 1

    # Retrieve normalization factors
    bkg_norm = ws3.var("CMS_hgg_0_2016_13TeV_bkg_norm").getVal()
    z_norm   = ws3.var(f"model_Z_c{category}_norm").getVal()
    h_norm   = ws3.var(f"model_H_c{category}_norm").getVal()

    y_tot = y_bkg*bkg_norm + y_Z*z_norm

    # --- Normalize on fine grid ---
    y_tot_norm = normalize(y_tot, x, data_hist, category)
    scale = y_tot_norm / y_tot  # consistent scaling
    y_bkg_norm = y_bkg*scale*bkg_norm / (y_bkg*bkg_norm + y_Z*z_norm) * rebin_factor
    y_Z_norm   = y_Z*scale*z_norm / (y_bkg*bkg_norm + y_Z*z_norm)*rebin_factor  # scale Z by its fraction in total
    y_H_norm = y_H*h_norm

    # Plot PDF on top plot
    label = pdf_names[i].replace("env_pdf_", "").replace(f"_cat{category}", "").replace("_"," ") if i < len(pdf_names) else f"pdf {i}"
    ax[0].plot(x, y_bkg_norm, label=label,  linewidth=5, alpha=0.3)
    

    # --- If this is the best PDF, also make the Data - nonRes plot ---
    if i == bestPdfIndex:
        print("we best pdf")
        #ax[0].plot(x, y_bkg_norm*rebin_factor, label=label,  linewidth=5, alpha=0.3, color='red')
        # Evaluate components on rebinned data grid
        y_bkg_data = []
        #y_Z_data   = []
        #y_H_data = []
        for val in x_data:
            mass.setVal(val)
            y_bkg_data.append(ws3.pdf("CMS_hgg_0_2016_13TeV_bkg_noZ").getVal())
            #y_Z_data.append(ws3.pdf(f"model_Z_c{category}").getVal())
            #y_H_data.append(ws3.pdf(f"model_H_c{category}").getVal())
        y_bkg_data = np.array(y_bkg_data)*bkg_norm
        #y_Z_data   = np.array(y_Z_data)*z_norm
        #y_H_data = np.array(y_H_data)*h_norm

        # Interpolate scaling from fine grid
        scale_data = np.interp(x_data, x, scale)
        y_bkg_data_norm = y_bkg_data * scale_data
        #y_Z_data_norm   = y_Z_data   * scale_data
        #y_H_data_norm  = y_H_data   * scale_data

        # Subtract non-resonant background
        y_subtracted = y_data - y_bkg_data_norm*rebin_factor
        ax[1].errorbar(x_data[(x_data<110) | (x_data>140) ], y_subtracted[(x_data<110) | (x_data>140) ], yerr=y_err[(x_data<110) | (x_data>140) ], fmt='o', color='black')

        # Optional: overlay Z contribution for reference
        #print("y_Z_data_norm", y_Z_data_norm)
        ax[1].plot(x, y_Z_norm, color='red', label="Z(bb)")
        ax[1].plot(x, y_H_norm*rebin_factor * 10, color='blue', label="H(bb) x 10")

        #print(y_H_data_norm*rebin_factor*10)


        # Horizontal zero line
        
        ax[1].hlines(y=0, xmin=x_data[0]-10, xmax=x_data[-1]+10, colors='red',  linewidth=5, alpha=0.3)
        ax[0].plot(x, (y_bkg_norm+y_Z_norm)*rebin_factor, label=label+"+Z(bb)", linewidth=1, color='red')

# Labels and legend
ax[1].set_xlabel(r"m$_{jj}$ [GeV]")
ax[0].set_ylabel("Events")
hep.cms.label(label="Private Work", data=True, lumi=41.6, ax=ax[0])
ax[1].set_ylabel("Residuals")
ax[0].legend(ncol=1, fontsize=18)
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*1.4)
ax[1].legend(ncol=2, fontsize=18, loc="upper right")

# Save figure
fig.savefig(f"/t3home/gcelotto/ggHbb/documentation/plotScripts/categories_fit/multipdf_cat{category}_best.png",
            bbox_inches='tight')