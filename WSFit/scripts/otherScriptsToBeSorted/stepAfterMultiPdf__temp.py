import ROOT
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import warnings
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', type=int, default="0", help='Number of category')
if hasattr(sys, 'ps1') or not sys.argv[1:]:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# Load ROOT file
file_path = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws%d.root"%args.category
f = ROOT.TFile.Open(file_path)
ws = f.Get("ws3")

# Access dataset
data = ws.data("rooHist_data_cat%d"%args.category)
higgs_hist = ws.data("rooHist_H_cat%d"%args.category)
z_hist = ws.data("rooHist_Z_cat%d"%args.category)
higgs_y = []
z_y = []
# Get the variable
mjj_var = ws.var("dijet_mass_c%d"%args.category)
xmin = mjj_var.getMin()
xmax = mjj_var.getMax()

# Convert dataset to arrays for plotting
data_x, data_y, bin_widths = [], [], []

for i in range(data.numEntries()):
    # Get the i-th bin (RooDataHist) as RooArgSet
    data_entry = data.get(i)
    higgs_entry = higgs_hist.get(i)

    # Get the value of the variable
    val = data_entry.getRealValue(mjj_var.GetName())
    data_x.append(val)

    # Get the weight (content of the bin)
    data_y.append(data.weight(i))        # note: weight(i) for RooDataHist
    higgs_y.append(higgs_hist.weight(i))
    z_y.append(z_hist.weight(i))
# Estimate bin width from consecutive points (assumes uniform bins)
if len(data_x) > 1:
    bin_width = data_x[1] - data_x[0]
else:
    bin_width = 1.0  # fallback

nbins = len(data_y)
print("[INFO] Binwidth is ", bin_width)

all_vars = ws.allVars()

# Filter _bkg_norm variables and extract type
bkg_norm_types = []
turnOn = 0
for var in all_vars:
    name = var.GetName()
    if name.endswith("_bkg_norm"):
        # Remove prefix 'env_pdf_' and suffix '_bkg_norm'
        type_name = name[len("env_pdf_") : -len("_bkg_norm")]
        print("Found ", name)
        bkg_norm_types.append(type_name)
    if name.endswith("_turnon_beta"):
        turnOn = 2

# Print result
data_x, data_y = np.array(data_x), np.array(data_y)
mask1 = data_x<105
mask2 = data_x>140
mask = (mask1) | (mask2)
plt.figure(figsize=(8,6))
plt.errorbar(data_x[mask], data_y[mask], yerr=np.sqrt(data_y)[mask], fmt='o', label='Data', color='black', zorder=1)
print("[INFO] These are the available bkg_norm_types", bkg_norm_types)

map_name_reduced = {
    "Bernstein":"bern",
    "Exponential":"exp",
    "PowerLaw":"pow"
}
n_events = sum(data_y)

background_functions = {}
for bkg_norm_type in bkg_norm_types:
    print("\n"*5)
    print("*"*30)
    input("Next")
    print("Function : ", bkg_norm_type)
    familyName, order = bkg_norm_type.split("_")
    familyShortName = map_name_reduced[familyName]


    norm_bkg = ws.var("env_pdf_%s_bkg_norm"%(bkg_norm_type)).getVal()
    norm_z   = ws.var("env_pdf_%s_z_norm"%(bkg_norm_type)).getVal()
    print("norm_z : ", norm_z)
    print("norm_bkg : ", norm_bkg)

    # Get PDFs
    pdf_bkg = ws.function(f"env_pdf_{familyName}_{order}_with_z")
    print(f"env_pdf_{familyName}_{order}_with_z")
    pdf_z   = ws.function("model_Z_c%d"%args.category)
    if pdf_bkg is None:
        print(f"Function env_pdf_{bkg_norm_type}_{familyShortName}{order} not found in workspace!")
    else:
        print(f"Function {pdf_bkg.GetName()} found.")
    
    #total pdf
    total_pdf_name = f"env_pdf_{familyName}_{order}_with_z"
    total_pdf = ws.pdf(total_pdf_name)
    

    # Evaluate combined PDF
    x_vals = np.linspace(xmin, xmax, nbins)
    bkgPlusZ = []
    z_vals = []
    qcd_vals = []
    mjj_var = ws.var("dijet_mass_c%d"%args.category)
    for x in x_vals:
        mjj_var.setVal(x)
        #y = (norm_bkg * pdf_bkg.getVal(ROOT.RooArgSet(var)) +
        #    norm_z * pdf_z.getVal(ROOT.RooArgSet(var))) 
        z =  pdf_z.getVal(ROOT.RooArgSet(mjj_var))
        z_vals.append(z)
        qcd =  pdf_bkg.getVal(ROOT.RooArgSet(mjj_var))
        qcd_vals.append(qcd)
        y = total_pdf.getVal(ROOT.RooArgSet(mjj_var)) * n_events * bin_width
        bkgPlusZ.append(y)

    # Combined PDF: higher zorder → drawn on top
    z_vals = np.array(z_vals)
    qcd_vals = np.array(qcd_vals)
    bkgPlusZ = np.array(bkgPlusZ)
    background_functions[familyName+order] = bkgPlusZ
    #area = np.trapz(bkgPlusZ[mask1], x_vals[mask1]) + np.trapz(bkgPlusZ[mask2], x_vals[mask2])
    #bkgPlusZ = bkgPlusZ/area * (norm_bkg + norm_z)*bin_width
    #area_qcd = np.sum(qcd_vals[mask1]*bin_width) + np.sum(qcd_vals[mask2]*bin_width) 
    #area_z = np.sum(z_vals[mask1]*bin_width) + np.sum(z_vals[mask2]*bin_width)
    area_qcd = np.sum(qcd_vals*bin_width)
    area_z = np.sum(z_vals*bin_width) 
    #area_qcd = np.trapz(qcd_vals[mask], x_vals[mask])
    #area_z = np.trapz(z_vals[mask], x_vals[mask]) 
    z_vals = z_vals/area_z
    qcd_vals = qcd_vals/area_qcd

    #bkgPlusZ = (qcd_vals * norm_bkg + z_vals * norm_z) * bin_width


    obs = data_y[mask]
    exp = bkgPlusZ[mask]

    # Avoid log(0) issues
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(obs > 0, obs * np.log(obs / exp), 0)
    chi2_poisson = 2 * np.sum(exp - obs + term)
    print("Chi2 : ", chi2_poisson)
    print("Nbins : ", np.sum(mask))
    print("Order : ", order)
    print("TurnOn : ", turnOn)
    print("Normalization : ", 2) # (one for total one for fraction)
    ndof = np.sum(mask) - int(order) -turnOn -2
    print("Ndof : ", ndof)
    print("Chi2/Ndof : ", chi2_poisson/ndof)
    plt.plot(x_vals, bkgPlusZ, label=(familyName+str(order)), lw=2, zorder=2)
    plt.xlabel("dijet_mass_c0")
    plt.ylabel("Events")
    plt.legend()
    plt.title("Data and Combined PDF")
    plt.savefig("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/combined_pdf_%d.png"%args.category)
    plt.show()

fig, ax = plt.subplots(1, 1)
all_values = np.array([background_functions[key] for key in background_functions])
avg_values = np.mean(all_values, axis=0)

# Plot each function relative to the average
fig, ax = plt.subplots(1, 1, figsize=(8,5))
for key in background_functions:
    relative = background_functions[key] - avg_values
    ax.plot(data_x, relative, label=key)
ax.errorbar(data_x, higgs_y, fmt='o', label='Higgs', color='black', zorder=1)
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel("Observable")
ax.set_ylabel("Deviation from mean")
ax.legend()
ax.set_title("Background functions relative to their average")
fig.savefig("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/differences.png", bbox_inches='tight')
