import ROOT
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)
# Open workspace
f = ROOT.TFile("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_2.root")
ws = f.Get("multipdf")  # replace with your workspace name

# Retrieve objects
pdf = ws.pdf("env_pdf_Bernstein_2_with_z")          # PDF
data = ws.data("rooHist_data_cat2")       # RooDataSet
mass = ws.var("dijet_mass_c2")        # observable

# Define range
blind_lowL = 105
blind_highL = 140
mass.setRange("blind_low", mass.getMin(), blind_lowL)
mass.setRange("blind_high", blind_highL, mass.getMax())

# Retrieve normalization parameters
bkg_norm = ws.var("env_pdf_Bernstein_2_bkg_norm").getVal()
z_norm   = ws.var("env_pdf_Bernstein_2_z_norm").getVal()
fitIntegral = pdf.createIntegral(mass, ROOT.RooFit.NormSet(mass), ROOT.RooFit.Range("blind_low,blind_high")).getVal()
Nvis = (bkg_norm + z_norm) * fitIntegral


# Make bins for plotting
nbins = 251
x_vals = np.linspace(mass.getMin(), mass.getMax(), nbins+1)
x_centers = 0.5*(x_vals[:-1] + x_vals[1:])

# Extract binned data


# Extract bin info
data_x, data_y = [], []
for i in range(data.numEntries()):
    val = data.get(i).getRealValue(mass.GetName())
    weight = data.weight()
    data_x.append(val)
    data_y.append(weight)



# Evaluate PDF at bin centers
y_vals = []
for mass_val in data_x:
    mass.setVal(mass_val)
    y=pdf.getVal(ROOT.RooArgSet(mass)) * int(Nvis) * (data_x[1]-data_x[0]) / fitIntegral
    y_vals.append(y)
y_pdf = np.array(y_vals)
print("binwidth", data_x[1]-data_x[0])
print("Integral is ", fitIntegral)
print("Nivs ", Nvis)
print("normval", (bkg_norm + z_norm))

data_x, data_y, y_vals = np.array(data_x), np.array(data_y), np.array(y_vals) 
mask = (data_x<105) | (data_x>140)
chi2 = np.sum((y_pdf[mask] - data_y[mask])**2/data_y[mask])
print(chi2)
ndof = np.sum(mask)  - 2 - 2 - 2
print(ndof)
print(chi2/ndof)


mass.setVal(90)
y=pdf.getVal(ROOT.RooArgSet(mass)) * int(Nvis) * (data_x[1]-data_x[0]) / fitIntegral
print("Function at 90 is ", y)
print("Raw Function at 90 is ", pdf.getVal(ROOT.RooArgSet(mass)))

# Plot with matplotlib
plt.figure(figsize=(8,6))
plt.errorbar(data_x, data_y, yerr=np.sqrt(data_y), fmt='o', label="Data")
plt.plot(data_x, y_pdf, label="PDF", color='red')
plt.xlabel("Mass [GeV]")
plt.ylabel("Events")
plt.title("Fit result (blinded range)")
plt.legend()
plt.savefig("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/prova3.png")
