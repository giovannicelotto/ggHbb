import ROOT
import matplotlib.pyplot as plt
import numpy as np

# Load ROOT file
file_path = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_2.root"
f = ROOT.TFile.Open(file_path)
ws = f.Get("multipdf")

# Access RooDataHist
data = ws.data("rooHist_data_cat2")
var = ws.var("dijet_mass_c2")  # adjust for your workspace

# Extract bin info
data_x, data_y = [], []
for i in range(data.numEntries()):
    val = data.get(i).getRealValue(var.GetName())
    weight = data.weight()
    data_x.append(val)
    data_y.append(weight)

# Estimate bin width (from consecutive points)
bin_width = data_x[1] - data_x[0] if len(data_x) > 1 else 1.0

# Get the RooAddPdf directly
pdf_name = "env_pdf_Exponential_1_with_z"  # this is the RooAddPdf
pdf = ws.pdf(pdf_name)

# Evaluate the PDF at the bin centers
y_vals = []
n_events = sum(data_y)
for x in data_x:
    var.setVal(x)
    # getVal already accounts for the internal normalization of RooAddPdf
    y = pdf.getVal(ROOT.RooArgSet(var)) * n_events * bin_width
    y_vals.append(y)
data_x, data_y, y_vals = np.array(data_x), np.array(data_y), np.array(y_vals) 
mask = (data_x<105) | (data_x>140)
chi2 = np.sum((y_vals[mask] - data_y[mask])**2/data_y[mask])
print(chi2)
ndof = np.sum(mask)  - 2 - 2 -1
print(ndof)
print(chi2/ndof)
# Plot
plt.figure(figsize=(8,6))
plt.errorbar(data_x, data_y, yerr=np.sqrt(data_y), fmt='o', label='Data', color='black', zorder=1)
plt.step(data_x, y_vals, where='mid', label=pdf_name, color='red', lw=2, zorder=2)
plt.xlabel(var.GetName())
plt.ylabel("Events")
plt.legend()
plt.title("Data and RooAddPdf")
plt.savefig("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/combined_pdf2.png")
