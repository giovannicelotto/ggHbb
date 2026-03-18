# %%
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
from functions import loadMultiParquet_v2, getCommonFilters
# %%
# Load data
particle = "Z"
if particle=="Z":
    processes = [19,20,21,22,35]
else:
    processes = [37]
df = loadMultiParquet_v2(paths=processes,nMCs=-1,columns=None, returnFileNumberList=False, returnNumEventsTotal=False, filters=getCommonFilters(btagWP="T"))[0]
# %%

def invariant_mass(pt1, eta1, phi1, m1,
                   pt2, eta2, phi2, m2):
    # particle 1
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1  = np.sqrt(pt1**2 * np.cosh(eta1)**2 + m1**2)

    # particle 2
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2  = np.sqrt(pt2**2 * np.cosh(eta2)**2 + m2**2)

    # sum four-momenta
    E  = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_res = E**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.maximum(m2_res, 0.0))


df['dijet_mass_uncor'] = invariant_mass(df.jet1_pt_uncor, df.jet1_eta, df.jet1_phi, df.jet1_mass/df.jet1_bReg2018,
                                       df.jet2_pt_uncor, df.jet2_eta, df.jet2_phi, df.jet2_mass/df.jet2_bReg2018)
# %%
# Plot with errors
bins = np.linspace(60, 200, 61)
fig, ax = plt.subplots(1, 1)
c_cor = np.histogram(df.dijet_mass, bins=bins, density=False, weights=df.flat_weight)[0]
c_uncor = np.histogram(df.dijet_mass_uncor, bins=bins, density=False, weights=df.flat_weight)[0]

c_cor_err = np.sqrt(np.histogram(df.dijet_mass, bins=bins, density=False, weights=(df.flat_weight)**2)[0])
c_uncor_err = np.sqrt(np.histogram(df.dijet_mass_uncor, bins=bins, density=False, weights=(df.flat_weight)**2)[0])
ax.errorbar((bins[1:]+bins[:-1])/2, c_cor, xerr=(bins[1]-bins[0])/2, yerr=c_cor_err, fmt='o', label='With bReg', color='red')
ax.errorbar((bins[1:]+bins[:-1])/2, c_uncor, xerr=(bins[1]-bins[0])/2, yerr=c_uncor_err, fmt='o', label='With bReg', color='black')
# %%
import numpy as np
from iminuit import Minuit


def gauss(x, norm, mu, sigma):
    return norm * np.exp(-0.5 * ((x - mu) / sigma)**2)


def dscb(x, norm, mu, sigma, alphaL, nL, alphaR, nR):
    t = (x - mu) / sigma
    y = np.zeros_like(x)

    AL = (nL / abs(alphaL))**nL * np.exp(-alphaL**2 / 2)
    BL = nL / abs(alphaL) - abs(alphaL)
    AR = (nR / abs(alphaR))**nR * np.exp(-alphaR**2 / 2)
    BR = nR / abs(alphaR) - abs(alphaR)

    left  = t < -alphaL
    right = t >  alphaR
    core  = (~left) & (~right)

    y[left]  = norm * AL * (BL - t[left])**(-nL)
    y[right] = norm * AR * (BR + t[right])**(-nR)
    y[core]  = norm * np.exp(-0.5 * t[core]**2)

    return y


def chi2_spectrum(x, y, yerr):

    def chi2(norm_g, sigma_g,
             #norm_cb, sigma_cb, alphaL, nL, alphaR, nR,
             mu):

        model = (
            gauss(x, norm_g, mu, sigma_g) 
            #dscb(x, norm_cb, mu, sigma_cb, alphaL, nL, alphaR, nR)
        )
        return np.sum(((y - model) / yerr)**2)

    return chi2


x = 0.5 * (bins[1:] + bins[:-1])
if particle=="Z":
    mask_unc = (c_uncor_err > 0) & (x > 60) & (x < 110)
    mask_cor = (c_cor_err > 0) & (x > 65) & (x < 120)
else:
    mask_unc = (c_uncor_err > 0) & (x > 75) & (x < 150)
    mask_cor = (c_cor_err > 0) & (x > 90) & (x < 155)
x_unc = x[mask_unc]
y_unc = c_uncor[mask_unc]
e_unc = c_uncor_err[mask_unc]

x_cor = x[mask_cor]
y_cor = c_cor[mask_cor]
e_cor = c_cor_err[mask_cor]


chi2_unc = chi2_spectrum(x_unc, y_unc, e_unc)

m_unc = Minuit(
    chi2_unc,
    norm_g=max(y_unc),
    sigma_g=10,
    #norm_cb=max(y_unc),
    #sigma_cb=15,
    #alphaL=1.5, nL=2,
    #alphaR=1.5, nR=2,
    mu=np.mean(x_unc)
)

m_unc.limits["sigma_g"] = (1e-3, None)
#m_unc.limits["sigma_cb"] = (1e-3, None)
m_unc.migrad()


chi2_cor = chi2_spectrum(x_cor, y_cor, e_cor)

m_cor = Minuit(
    chi2_cor,
    norm_g=max(y_cor),
    sigma_g=8,
    #norm_cb=max(y_cor),
    #sigma_cb=12,
    #alphaL=1.5, nL=2,
    #alphaR=1.5, nR=2,
    mu=np.mean(x_cor)
)

m_cor.limits["sigma_g"] = (1e-3, None)
#m_cor.limits["sigma_cb"] = (1e-3, None)
m_cor.migrad()


def print_results(bins, m, name):
    chi2 = m.fval
    ndof = len(bins)-1 - len(m.values)
    print(f"{name}:")
    print(f"  chi2/ndof = {chi2:.1f} / {ndof}")
    print(f"  mean      = {m.values['mu']:.3f}")
    print(f"  sigma_g   = {m.values['sigma_g']:.3f}")
    #print(f"  sigma_cb  = {m.values['sigma_cb']:.3f}")
    print()

print_results(bins, m_unc, "Uncorrected (black)")
print_results(bins, m_cor, "Corrected (red)")

# %%
fig, ax = plt.subplots(1, 1)

ax.errorbar(x, c_uncor, yerr=c_uncor_err, fmt='o', color='black', label='Uncorrected')
ax.errorbar(x, c_cor, yerr=c_cor_err, fmt='o', color='red', label='With bReg')

xfit = np.linspace(bins[0], bins[-1], 500)

def eval_model(m, x):
    return (gauss(x, m.values['norm_g'], m.values['mu'], m.values['sigma_g']) 
            #dscb(x, m.values['norm_cb'], m.values['mu'], m.values['sigma_cb'],
            #     m.values['alphaL'], m.values['nL'],
            #     m.values['alphaR'], m.values['nR'])
            )

ax.plot(xfit, eval_model(m_unc, xfit), color='black')
ax.plot(xfit, eval_model(m_cor, xfit), color='red')
ax.text(0.05, 0.25+0.7, '$\mu = %.1f \pm %.1f$ GeV '%(m_unc.values["mu"], m_unc.errors["mu"]), transform=ax.transAxes, fontsize=18)
ax.text(0.05, 0.25+0.65, '$\sigma = %.1f \pm %.1f$ GeV'%(m_unc.values["sigma_g"], m_unc.errors["sigma_g"]), transform=ax.transAxes, fontsize=18)
ax.text(0.05, 0.25+0.6, '$\sigma/\mu = %.1f$ %%'%(m_unc.values["sigma_g"]/m_unc.values["mu"]*100), transform=ax.transAxes, fontsize=18)

ax.text(0.95, 0.25+0.7, '$\mu = %.1f \pm %.1f$ GeV'%(m_cor.values["mu"], m_cor.errors["mu"]), transform=ax.transAxes, fontsize=18, ha='right', color='red')
ax.text(0.95, 0.25+0.65, '$\sigma = %.1f \pm %.1f$ GeV'%(m_cor.values["sigma_g"], m_cor.errors["sigma_g"]), transform=ax.transAxes, fontsize=18, ha='right', color='red')
ax.text(0.95, 0.25+0.6, '$\sigma/\mu = %.1f$ %%'%(m_cor.values["sigma_g"]/m_cor.values["mu"]*100), transform=ax.transAxes, fontsize=18, ha='right', color='red')

ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.2)
ax.text(0.95, 0.2, '$p_T^{jj} > 100$ GeV\n$p_T^{\mu}$> 9 GeV\n |IPsig$^{\mu}$| > 6\n|$\eta^{\mu}$| < 1.5\n$p_T^j>20$ GeV', transform=ax.transAxes, fontsize=14, ha='right', color='black')
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]))
ax.legend()
hep.cms.label(data=False)
fig.savefig(f"/t3home/gcelotto/ggHbb/documentation/plots/bRegression/bRegEffect_on_dijetMass{particle}.png", bbox_inches='tight')
# %%
