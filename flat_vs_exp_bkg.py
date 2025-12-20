# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
# %%
# -------------------------------
# Parameters
# -------------------------------
np.random.seed(42)

# mass range
xmin, xmax = 50, 200
signal_mean, signal_sigma = 125, 14

# total events for exponential background
n_bkg_total = 5000
tau = 50.0  # slope of exp

# signal strength
n_signal = 800  # small signal

# -------------------------------
# Generate exponential background
# -------------------------------
def sample_exponential(n, tau, xmin, xmax):
    # inverse transform sampling for truncated exponential
    u = np.random.rand(n)
    return xmin - tau * np.log(1 - u * (1 - np.exp(-(xmax - xmin) / tau)))

bkg_exp = sample_exponential(n_bkg_total, tau, xmin, xmax)

# take only events in 100-150 for counting
window_mask = (bkg_exp > 100) & (bkg_exp < 150)
n_bkg_in_window = np.sum(window_mask)

# generate small Gaussian signal
signal = np.random.normal(signal_mean, signal_sigma, n_signal)

# combine to make data
data_exp = np.concatenate([bkg_exp, signal])
data_exp = data_exp[(data_exp > xmin) & (data_exp < xmax)]

# -------------------------------
# Define models
# -------------------------------
def model_exp(x, N, tau):
    return N * np.exp(-x / tau)

def model_exp_plus_gaus(x, N, tau, Ns, mu, sigma):
    return model_exp(x, N, tau) + Ns * norm.pdf(x, mu, sigma)

def model_flat(x, N):
    return N * np.ones_like(x)

def model_flat_plus_gaus(x, N, Ns, mu, sigma):
    return model_flat(x, N) + Ns * norm.pdf(x, mu, sigma)

# -------------------------------
# Negative Log-Likelihood
# -------------------------------
def nll_exp(params, x):
    N, tau = params
    pdf = np.exp(-x / tau)
    pdf /= np.trapz(pdf, x)  # normalize
    return -np.sum(np.log(pdf))

def nll_exp_plus_signal(params, x):
    N, tau, Ns = params
    pdf = np.exp(-x / tau)
    pdf = N * pdf / np.trapz(pdf, x)
    pdf += Ns * norm.pdf(x, signal_mean, signal_sigma)
    pdf /= np.trapz(pdf, x)
    return -np.sum(np.log(pdf))

def nll_flat(params, x):
    return -np.sum(np.log(np.ones_like(x) / (xmax - xmin)))

def nll_flat_plus_signal(params, x):
    N, Ns = params
    pdf = N * np.ones_like(x)
    pdf += Ns * norm.pdf(x, signal_mean, signal_sigma)
    pdf /= np.trapz(pdf, x)
    return -np.sum(np.log(pdf))




# %%









# %%
# -------------------------------
# Perform fits (exponential case)
# -------------------------------
res_bkg_only_exp = minimize(lambda p: nll_exp(p, data_exp), x0=[1.0, 5000.0], bounds=[(1e-6, None), (1e-6, None)])
res_bkg_plus_sig_exp = minimize(lambda p: nll_exp_plus_signal(p, data_exp),
                                x0=[1.0, 5000.0, n_signal],
                                bounds=[(1e-6, None), (1e-6, None), (0, None)])

nll1_exp = res_bkg_only_exp.fun
nll2_exp = res_bkg_plus_sig_exp.fun

significance_exp = np.sqrt(2 * (nll1_exp - nll2_exp))
# %%
# -------------------------------
# Generate flat background with same B in window
# -------------------------------
# compute expected density of flat background so that it gives same B in 100-150
flat_density = n_bkg_in_window / (150 - 100)
flat_total = int(flat_density * (xmax - xmin))

bkg_flat = np.random.uniform(xmin, xmax, flat_total)
data_flat = np.concatenate([bkg_flat, signal])
data_flat = data_flat[(data_flat > xmin) & (data_flat < xmax)]

# -------------------------------
# Perform fits (flat case)
# -------------------------------
res_bkg_only_flat = minimize(lambda p: nll_flat(p, data_flat), x0=[1.0], bounds=[(1e-6, None)])
res_bkg_plus_sig_flat = minimize(lambda p: nll_flat_plus_signal(p, data_flat),
                                 x0=[1.0,n_signal],
                                 bounds=[(1e-6, None), (1, None)])

nll1_flat = res_bkg_only_flat.fun
nll2_flat = res_bkg_plus_sig_flat.fun

significance_flat = np.sqrt(2 * (nll1_flat - nll2_flat))

# -------------------------------
# Results
# -------------------------------
print(f"Exp background significance:  {significance_exp:.3f}")
print(f"Flat background significance: {significance_flat:.3f}")

# %%
fig, ax  = plt.subplots(1, 1)
ax.hist(data_exp, bins=100, histtype='step')
ax.hist(data_flat, bins=100, histtype='step')

# %%
# -------------------------------
# Optional: plot
# -------------------------------
bins = np.linspace(xmin, xmax, 60)
x = 0.5 * (bins[1:] + bins[:-1])
counts_exp, _ = np.histogram(data_exp, bins=bins)
counts_flat, _ = np.histogram(data_flat, bins=bins)



plt.figure(figsize=(10, 6))

# Plot exponential data
plt.step(x, counts_exp, where='mid', label='Exp + Signal (data)', color='C0')
# Plot exponential fits
exp_params = res_bkg_plus_sig_exp.x
yfit_exp = model_exp_plus_gaus(x, *exp_params, mu=signal_mean, sigma=signal_sigma)
plt.plot(x, yfit_exp * (np.trapz(counts_exp, x) / np.trapz(yfit_exp, x)), 'C0--', label='Exp+Sig fit')

# Plot flat data
plt.step(x, counts_flat, where='mid', label='Flat + Signal (data)', color='C1')
# Plot flat fits
flat_params = res_bkg_plus_sig_flat.x
yfit_flat = model_flat_plus_gaus(x, *flat_params, mu=signal_mean, sigma=signal_sigma)
plt.plot(x, yfit_flat * (np.trapz(counts_flat, x) / np.trapz(yfit_flat, x)), 'C1--', label='Flat+Sig fit')

plt.xlabel("Mass")
plt.ylabel("Events")
plt.title("Toy Signal on Exponential vs Flat Background")
plt.legend()
plt.show()
# %%
