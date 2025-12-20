# %%
from functions import loadMultiParquet_v2, getCommonFilters
import matplotlib.pyplot as plt
import numpy as np
def compute_mode_fwhm(hist, bin_edges):
    """Compute mode and FWHM of a histogram."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mode_idx = np.argmax(hist)
    mode = bin_centers[mode_idx]

    half_max = hist[mode_idx] / 2.0
    above_half = np.where(hist >= half_max)[0]
    if len(above_half) > 1:
        fwhm = bin_centers[above_half[-1]] - bin_centers[above_half[0]]
    else:
        fwhm = np.nan
    return mode, fwhm

def invariant_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    # Convert to Cartesian components
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + pt1**2 * np.cosh(eta1)**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + pt2**2 * np.cosh(eta2)**2)

    # Combine
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    # Invariant mass
    mass = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))
    return mass

def getCoefficientsFromFit(df, response, variable_x, min_x, max_x, nbins, poly_degree, text=None):

    # Define bins in gen pt (you can adjust binning as needed)

    bins = np.linspace(min_x, max_x, nbins)  # 10 GeV bins up to 400
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute response
    #response = 

    # Digitize: assign each event to a pt_gen bin
    bin_indices = np.digitize(df[variable_x], bins) - 1
    from statistics import median
    # Compute mean and std of response in each bin
    means = [median(response[bin_indices == i]) for i in range(len(bins) - 1)]
    stds = [response[bin_indices == i].std() for i in range(len(bins) - 1)]
    counts = [np.sum(bin_indices == i) for i in range(len(bins) - 1)]

    # Convert to numpy arrays (helps with masking empty bins)
    means = np.array(means)
    stds = np.array(stds)
    counts = np.array(counts)

    # Mask bins with too few entries (optional)
    mask = counts > 0
    x = bin_centers[mask]
    y = means[mask]
    yerr = stds[mask] / np.sqrt(counts[mask])


    # --- Fit (same as before) ---
    fit_order = poly_degree
    coeffs = np.polyfit(x, y, fit_order, w=1/yerr)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(min_x, max_x, 300)

    y_fit = poly(x_fit)

    # Compute residuals (normalized)
    residuals = (y - poly(x)) / yerr

    # --- Create figure with 2 subplots (main + residuals) ---
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # --- Main plot ---
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='Jet response', color='black')
    ax.plot(x_fit, y_fit, '-', color='orange', lw=2,
            label=f'Polynomial (order {fit_order}) fit')
    ax.axhline(1, color='r', linestyle='--', label='Perfect response')

    ax.set_ylabel(r'$p_T^{gen} / p_T^{reco}$')
    #ax.set_title('Jet1 Response vs Gen Jet $p_T$')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0.5, 1.5)

    # --- Residuals plot ---
    ax_res.errorbar(x, residuals, yerr=1, fmt='o', capsize=3, color='black')
    ax_res.axhline(0, color='r', linestyle='--')
    ax_res.set_xlabel(variable_x)
    ax_res.set_ylabel('Residuals')
    ax_res.grid(True)
    ax_res.set_ylim(-3, 3)
    if text is not None:
        ax.text(x=0.95, y=0.95, s=text, fontsize=12, ha='right', transform=ax.transAxes)
    else:
        pass
    plt.tight_layout()
    plt.show()

    # --- Optionally print coefficients ---
    print(f"Polynomial coefficients (highest degree first):\n{coeffs}")
    return poly

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return dphi

def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)



def getCoefficientsFromFit2D(df, response, variable_x, variable_y,
                             min_x, max_x, nbins_x,
                             min_y, max_y, nbins_y,
                             poly_degree):
    """
    Fit response vs variable_x (e.g. mass) in bins of variable_y (e.g. eta),
    returning one polynomial per bin of variable_y.
    """
    # Define bins
    bins_y = np.linspace(min_y, max_y, nbins_y + 1)
    bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    
    polynomials = []
    valid_bins = []

    for i in range(len(bins_y) - 1):
        # Select entries in current y-bin
        mask = (df[variable_y] >= bins_y[i]) & (df[variable_y] < bins_y[i + 1])
        if np.sum(mask) < 20:
            continue  # skip bins with too few entries

        sub_df = df[mask]
        print("sub_df with length %d"%(len(sub_df)))
        sub_response = response[mask]

        # Fit response vs variable_x for this slice
        poly = getCoefficientsFromFit(
            sub_df,
            sub_response,
            variable_x,
            min_x,
            max_x,
            nbins_x,
            poly_degree,
            text=  f"{bins_y[i]:.1f} <= "+variable_y+f" < {bins_y[i + 1]:.1f}")

        polynomials.append(poly)
        valid_bins.append((bins_y[i], bins_y[i+1]))

    return polynomials, valid_bins
def apply_2d_poly_correction(pt_toBeCorrected, variable_x, binned_variable, polys, variable_bins):
    corrected_pt = np.zeros_like(pt_toBeCorrected)
    for i, (eta_min, eta_max) in enumerate(variable_bins):
        mask = (binned_variable >= eta_min) & (binned_variable < eta_max)
        corrected_pt[mask] = pt_toBeCorrected[mask] * polys[i](variable_x[mask])
    return corrected_pt
# %%
dfs, sumw = loadMultiParquet_v2(paths=[37], nMCs=-1, columns=None, returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters(btagTight=True), training=False, isJEC=0)
# %%
df=dfs[0]
for f in df.columns:
    print(f)
# %%
fig, ax = plt.subplots(1, 1)
bins_mass = np.linspace(40, 180, 101)
ax.hist(invariant_mass(df.jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass, df.jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values, bins=bins_mass, histtype='step', label="Invariant mass (Reco)")

#df_close = df[((df.jet1_pt - df.jet1_genQuark_pt) < 15) & ((df.jet2_pt - df.jet2_genQuark_pt)< 15)]
#ax.hist(invariant_mass(df.jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass, df.jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values, bins=bins_mass, histtype='step', label="Invariant mass (Reco)")
#ax.hist(invariant_mass(df_close.jet1_pt, df_close.jet1_eta, df_close.jet1_phi, df_close.jet1_mass, df_close.jet2_pt, df_close.jet2_eta, df_close.jet2_phi, df_close.jet2_mass).values, bins=bins_mass, histtype='step', label="Jets pt similar to Quark pT (10%)")
#ax.hist(invariant_mass(df_close.jet1_genQuark_pt, df_close.jet1_eta, df_close.jet1_phi, df_close.jet1_mass, df_close.jet2_genQuark_pt, df_close.jet2_eta, df_close.jet2_phi, df_close.jet2_mass).values, bins=bins_mass, histtype='step', label="Jets pt similar to Quark pT using Quark pT")
ax.hist(invariant_mass(df.jet1_genQuark_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass, df.jet2_genQuark_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values, bins=bins_mass, histtype='step', label="Invariant mass using Quark pT")
ax.legend(bbox_to_anchor=(1, 1))
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# --- Load your dataframe ---
# df = pd.read_csv("your_file.csv")

# --- Define inputs and target ---


# --- Define features and targets ---
features = [
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", "jet1_leadTrackPt",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass", "jet2_leadTrackPt",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", "jet3_leadTrackPt",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass", "jet4_leadTrackPt",
    "muon_pt", "muon_eta", "muon_phi", 
    "muon2_pt", "muon2_eta", "muon2_phi", 
    "nJets", "dijet_pTAsymmetry"
]
target1 = "jet1_pt_over_genQuark_pt"
target2 = "jet2_pt_over_genQuark_pt"

# --- Create targets ---
df["jet1_pt_over_genQuark_pt"] = df["jet1_pt"] / df["jet1_genQuark_pt"]
df["jet2_pt_over_genQuark_pt"] = df["jet2_pt"] / df["jet2_genQuark_pt"]
df = df[(df.jet1_pt_over_genQuark_pt<3) & (df.jet2_pt_over_genQuark_pt<3)]
# --- Split data manually ---
X = df[features].values
y1 = df[target1].values
y2 = df[target2].values

N_train = len(df)//2
X_train, X_test = X[:N_train], X[N_train:]
y1_train, y1_test = y1[:N_train], y1[N_train:]
y2_train, y2_test = y2[:N_train], y2[N_train:]

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Convert to torch tensors ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y1_train = torch.tensor(y1_train, dtype=torch.float32).unsqueeze(1)
y2_train = torch.tensor(y2_train, dtype=torch.float32).unsqueeze(1)
y1_test = torch.tensor(y1_test, dtype=torch.float32).unsqueeze(1)
y2_test = torch.tensor(y2_test, dtype=torch.float32).unsqueeze(1)

# --- Define a simple neural network with two outputs ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
            nn.ReLU(),
            nn.Linear(6, 2)  # two outputs: jet1_ratio, jet2_ratio
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN(input_dim=X_train_tensor.shape[1])
# %%
# --- Define custom loss: sum of two MSEs ---
criterion = nn.MSELoss()

def double_mse_loss(pred, y1, y2):
    loss1 = criterion(pred[:, 0], y1.squeeze())
    loss2 = criterion(pred[:, 1], y2.squeeze())
    return loss1 + loss2

optimizer = optim.Adam(model.parameters(), lr=1e-2)

# --- Training loop ---
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = double_mse_loss(preds, y1_train, y2_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    preds_test = model(X_test_tensor)
    test_loss = double_mse_loss(preds_test, y1_test, y2_test)
print(f"Test loss (sum of MSEs): {test_loss.item():.6f}")

# %%
from scipy.stats import norm
from scipy.optimize import curve_fit
import mplhep as hep
hep.style.use("CMS")
def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
fig, ax = plt.subplots(1, 1, figsize=(8,6))
bins_mass = np.linspace(60, 200, 101)
mass_noncorr = invariant_mass(df.jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass, df.jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
mass_gen = invariant_mass(df.jet1_genQuark_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass, df.jet2_genQuark_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
mass_NN = invariant_mass(df[N_train:].jet1_pt * 1/ model(torch.tensor(scaler.transform(df[features].values[N_train:]), dtype=torch.float32)).squeeze(1).detach().numpy()[:,0],
                       df[N_train:].jet1_eta, df[N_train:].jet1_phi, df[N_train:].jet1_mass,
                       df[N_train:].jet2_pt * 1/ model(torch.tensor(scaler.transform(df[features].values[N_train:]), dtype=torch.float32)).squeeze(1).detach().numpy()[:,1],
                       df[N_train:].jet2_eta, df[N_train:].jet2_phi, df[N_train:].jet2_mass).values

# Plot histograms and compute stats
for data, label, color in [
    (mass_noncorr, "Invariant mass (Reco)", "blue"),
    (mass_gen, "Invariant mass using Quark pT", "red"),
    (mass_NN, "Invariant mass (NN corrected)", "green")
]:
    hist, bin_edges = np.histogram(data, bins=bins_mass)
    mode, fwhm = compute_mode_fwhm(hist, bin_edges)
    
    c = ax.hist(data, bins=bins_mass, histtype='step', label=f"{label} (Mode={mode:.1f}, FWHM={fwhm/mode*100:.1f}%)", color=color, density=True)[0]
    centers = 0.5 * (bins_mass[1:] + bins_mass[:-1])
    try:
        popt, _ = curve_fit(gauss, centers, c, p0=[0.04, 125, 10])
        A, mu, sigma = popt
        x_fit = np.linspace(bins_mass[0], bins_mass[-1], 500)
        y_fit = gauss(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color=color,
                label=f"Fit {label}: μ={mu:.1f}, σ={sigma:.1f} (σ/μ={sigma/mu*100:.1f}%)")
    except RuntimeError:
        print(f"⚠️ Gaussian fit failed for {label}")

ax.set_xlabel("Invariant mass [GeV]")
ax.set_ylabel("Norm .Events")
ax.legend(fontsize=14)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.show()







# %%

poly_jet1 = getCoefficientsFromFit(df, df['jet1_genQuark_pt']/df['jet1_pt'] , 'muon_pt', 9, 70, 81, 5)
poly_jet2 = getCoefficientsFromFit(df, df['jet2_genQuark_pt']/df['jet2_pt'] , 'jet2_leadTrackPt', 5, 50, 81, 5, text="Jet2")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))


# Compute invariant masses
mass_noncorr = invariant_mass(df.jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                              df.jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
mass_gen = invariant_mass(df.jet1_genQuark_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                          df.jet2_genQuark_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
corr_jet1_pt = df.jet1_pt*poly_jet1(df.muon_pt)
corr_jet2_pt = df.jet2_pt*poly_jet2(df.jet2_leadTrackPt)
mass_corr = invariant_mass(corr_jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                           corr_jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values

# Plot histograms and compute stats
for data, label, color in [
    (mass_noncorr, "Non corrected", "C0"),
    (mass_gen, "Gen Level", "C1"),
    (mass_corr, "Corrected", "C2")
]:
    hist, bin_edges = np.histogram(data, bins=bins_mass)
    mode, fwhm = compute_mode_fwhm(hist, bin_edges)
    c=ax.hist(data, bins=bins_mass, histtype='step', label=f"{label} (Mode={mode:.1f}, FWHM={fwhm/mode*100:.1f}%)", color=color)[0]
    try:
        popt, _ = curve_fit(gauss, centers, c, p0=[0.04, 125, 10])
        A, mu, sigma = popt
        x_fit = np.linspace(bins_mass[0], bins_mass[-1], 500)
        y_fit = gauss(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color=color,
                label=f"{label} fit: μ={mu:.1f}, σ={sigma:.1f} (σ/μ={sigma/mu*100:.1f}%)")
    except RuntimeError:
        print(f"⚠️ Gaussian fit failed for {label}")

ax.set_xlabel("Invariant mass [GeV]")
ax.set_ylabel("Events")
ax.legend(bbox_to_anchor=(1, 1))








# %%

# ΔR between each gen b and each jet
for i in range(1, 5):
    df[f'dR_bgen_jet{i}'] = delta_r(df['b_gen_eta'], df['b_gen_phi'], df[f'jet{i}_eta'], df[f'jet{i}_phi'])
    df[f'dR_antibgen_jet{i}'] = delta_r(df['antib_gen_eta'], df['antib_gen_phi'], df[f'jet{i}_eta'], df[f'jet{i}_phi'])

# %%
df['bgen_bestjet'] = df[[f'dR_bgen_jet{i}' for i in range(1, 5)]].idxmin(axis=1)
df['antibgen_bestjet'] = df[[f'dR_antibgen_jet{i}' for i in range(1, 5)]].idxmin(axis=1)
df['bgen_bestjet'] = df['bgen_bestjet'].str.extract('(\d)').astype(int)
df['antibgen_bestjet'] = df['antibgen_bestjet'].str.extract('(\d)').astype(int)
print("Best jet for b_gen:")
print(df['bgen_bestjet'].value_counts().sort_index())

print("\nBest jet for antib_gen:")
print(df['antibgen_bestjet'].value_counts().sort_index())

# %%



# %%

# %%


# %%














# %%
# %%
df = df[df.muon_pt<70]
df = df[df.jet2_leadTrackPt<70]
# %%
from statistics import median
bins  = np.linspace(0, 2, 101)
fig, ax = plt.subplots(1, 1)
ax.hist(np.clip((df.jet1_genQuark_pt)/(df.jet1_pt), bins[0], bins[-1]), bins=bins)
ax.vlines(x=median(df.jet1_genQuark_pt/(df.jet1_pt)), ymin=0, ymax=1000, color='red', label='median')
ax.vlines(x=np.mean(df.jet1_genQuark_pt/(df.jet1_pt)), ymin=0, ymax=1000, color='orange', label='mean')
ax.legend()
# %%

# For jet1:
polys_jet1, eta_bins_jet1 = getCoefficientsFromFit2D(
    df,
    df['jet1_genQuark_pt'] / df['jet1_pt'],
    variable_x='muon_pt',
    variable_y='jet1_eta',
    min_x=9, max_x=70, nbins_x=81,
    min_y=-1.5, max_y=1.5, nbins_y=3,
    poly_degree=2
)
# %%
# For jet2:
polys_jet2, eta_bins_jet2 = getCoefficientsFromFit2D(
    df,
    df['jet2_genQuark_pt'] / df['jet2_pt'],
    variable_x='jet2_leadTrackPt',
    variable_y='jet2_eta',
    min_x=5, max_x=70, nbins_x=51,
    min_y=-2.5, max_y=2.5, nbins_y=3,
    poly_degree=3
)
# %%
corr_jet1_pt = apply_2d_poly_correction(pt_toBeCorrected=df['jet1_pt'],
                                        variable_x=df['muon_pt'],
                                        binned_variable=df['jet1_eta'],
                                        polys=polys_jet1,
                                        variable_bins=eta_bins_jet1)
corr_jet2_pt = apply_2d_poly_correction(df['jet2_pt'],
                                        df['jet2_leadTrackPt'], 
                                        df['jet2_eta'],
                                        polys_jet2,
                                        eta_bins_jet2)



# %%
# --- Mass distributions ---
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
bins_mass = np.linspace(10, 200, 201)

# Compute invariant masses
mass_noncorr = invariant_mass(df.jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                              df.jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
mass_gen = invariant_mass(df.jet1_genQuark_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                          df.jet2_genQuark_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values
mass_corr = invariant_mass(corr_jet1_pt, df.jet1_eta, df.jet1_phi, df.jet1_mass,
                           corr_jet2_pt, df.jet2_eta, df.jet2_phi, df.jet2_mass).values


# Plot histograms and compute stats
for data, label, color in [
    (mass_noncorr, "Non corrected", "C0"),
    (mass_gen, "Gen Level", "C1"),
    (mass_corr, "Corrected", "C2")
]:
    hist, bin_edges = np.histogram(data, bins=bins_mass)
    mode, fwhm = compute_mode_fwhm(hist, bin_edges)
    c=ax.hist(data, bins=bins_mass, histtype='step', label=f"{label} (Mode={mode:.1f}, FWHM={fwhm/mode*100:.1f}%)", color=color)[0]
    try:
        popt, _ = curve_fit(gauss, (bins_mass[1:] + bins_mass[:-1])/2, c, p0=[0.04, 125, 10])
        A, mu, sigma = popt
        x_fit = np.linspace(bins_mass[0], bins_mass[-1], 500)
        y_fit = gauss(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color=color,
                label=f"{label} fit: μ={mu:.1f}, σ={sigma:.1f} (σ/μ={sigma/mu*100:.1f}%)")
    except RuntimeError:
        print(f"⚠️ Gaussian fit failed for {label}")

ax.set_xlabel("Invariant mass [GeV]")
ax.set_ylabel("Events")
ax.legend(bbox_to_anchor=(1, 1))
# %%




import numpy as np
import matplotlib.pyplot as plt

# Extract quantities
gen_pt = df['jet1_genQuark_pt'].values
reco_pt = df['jet1_pt'].values

# Define binning (adjust as needed)
pt_min, pt_max = 0, np.percentile(gen_pt, 99.5)  # avoid outliers
bins = np.linspace(pt_min, pt_max, 100)

# --- 2D histogram ---
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
h = ax.hist2d(gen_pt, reco_pt, bins=[bins, bins], cmap='viridis', norm='log')

# --- Reference line (perfect reconstruction) ---
ax.plot([pt_min, pt_max], [pt_min, pt_max], 'r--', lw=2, label='Perfect match')

# --- Labels and colorbar ---
ax.set_xlabel(r'$p_T^{gen}$ [GeV]')
ax.set_ylabel(r'$p_T^{reco}$ [GeV]')
ax.set_title('Jet1: Reconstructed vs Generated $p_T$')
ax.legend()
fig.colorbar(h[3], ax=ax, label='Events')

ax.grid(True)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Extract quantities
gen_pt = df['jet2_genQuark_pt'].values
reco_pt = df['jet2_pt'].values

# Define binning (adjust as needed)
pt_min, pt_max = 0, np.percentile(gen_pt, 99.5)  # avoid outliers
bins = np.linspace(pt_min, pt_max, 100)

# --- 2D histogram ---
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
h = ax.hist2d(gen_pt, reco_pt, bins=[bins, bins], cmap='viridis', norm='log')

# --- Reference line (perfect reconstruction) ---
ax.plot([pt_min, pt_max], [pt_min, pt_max], 'r--', lw=2, label='Perfect match')

# --- Labels and colorbar ---
ax.set_xlabel(r'$p_T^{gen}$ [GeV]')
ax.set_ylabel(r'$p_T^{reco}$ [GeV]')
ax.set_title('Jet2: Reconstructed vs Generated $p_T$')
ax.legend()
fig.colorbar(h[3], ax=ax, label='Events')

ax.grid(True)
plt.tight_layout()
plt.show()

# %%
