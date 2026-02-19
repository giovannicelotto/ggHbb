# %%
from scipy.interpolate import griddata
import numpy as np
import uproot 
import mplhep as hep
hep.style.use("CMS")
# %%
file_name = "/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombine.scan2D.MultiDimFit.mH125.root"
f = uproot.open(file_name)
t = f["limit"]
branches = t.arrays()
r = branches["r"]
rateZbb = branches["rateZbb"]
deltaNLL = branches["deltaNLL"]
# %%


r = np.asarray(r)
rateZbb = np.asarray(rateZbb)
deltaNLL = np.asarray(2*deltaNLL)
# %%
nr = 200
nz = 200

r_grid = np.linspace(r.min(), r.max(), nr)
z_grid = np.linspace(rateZbb.min(), rateZbb.max(), nz)

R, Z = np.meshgrid(r_grid, z_grid)
# %%
from scipy.interpolate import griddata

points = np.column_stack((r[1:], rateZbb[1:]))

deltaNLL_grid = griddata(
    points,
    deltaNLL[1:],
    (R, Z),
    method="linear"
)
# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

fig, ax = plt.subplots(1, 1)

# Filled background
cf = ax.contourf(
    R,
    Z,
    deltaNLL_grid,
    levels=50,
    cmap='YlGnBu'
)

# 1σ and 2σ contours
c1 = ax.contour(
    R,
    Z,
    deltaNLL_grid,
    levels=[2.30],
    linewidths=2,
    colors='red'
)

c2 = ax.contour(R, Z, deltaNLL_grid,
    levels=[5.99], linewidths=2, linestyles="--", colors='red'
)
print(r[np.argmin(deltaNLL)], rateZbb[np.argmin(deltaNLL)])
sm = ax.scatter(r[np.argmin(deltaNLL)], rateZbb[np.argmin(deltaNLL)] , marker='*', color='red', s=100, label='SM')
line1 = mlines.Line2D([], [], color='red', linewidth=2, label='1σ CL')
line2 = mlines.Line2D([], [], color='red', linewidth=2, linestyle='--', label='2σ CL')
ax.legend(handles=[line1, line2, sm], loc='upper right')
# Colorbar
fig.colorbar(cf, ax=ax, label=r"2$\Delta \mathrm{NLL}$")

ax.set_xlabel(f"$r_H$")
ax.set_ylabel(f"$r_Z$")
#x.set_xlim(-6,8)
#ax.set_ylim(0.5, 1.5)

outName = "/t3home/gcelotto/ggHbb/WSFit/output/combined/2Dscan_rateZbb_r.png"
print("Saving plot to ", outName)
fig.savefig(outName, bbox_inches='tight')
# %%


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
mask1D = (rateZbb==1)
# masked data
x = r[mask1D]
y = deltaNLL[mask1D]
y = y - np.min(y)

# sort (important for interpolation & root finding)
order = np.argsort(x)
x = x[order]
y = y[order]

# interpolation
interp = interp1d(x, y, kind="cubic")

# dense grid for plotting
x_dense = np.linspace(x.min(), x.max(), 1000)
y_dense = interp(x_dense)

# find minimum
xmin = x_dense[np.argmin(y_dense)]

# find intersections with NLL = 1
roots = []
for i in range(len(x) - 1):
    if (y[i] - 1) * (y[i+1] - 1) < 0:
        root = brentq(lambda xx: interp(xx) - 1, x[i], x[i+1])
        roots.append(root)

# plot
fig, ax = plt.subplots(1, 1)
ax.plot(x_dense, y_dense, label=r"Interpolated $\Delta$NLL")
ax.scatter(x, y, color="black", zorder=3, label="Scan points")

ax.axhline(1.0, linestyle="--", color="red", label=r"$\Delta$NLL = 1")
ax.axvline(xmin, linestyle=":", color="gray")

for r0 in roots:
    ax.axvline(r0, linestyle="--", color="blue")
    ax.scatter(r0, 1.0, color="blue", zorder=4)

ax.set_xlabel(r"$r_H$")
ax.set_ylabel(r"$\Delta NLL$")
ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/WSFit/output/combined/1Dscan_rH_sliceZ.png", bbox_inches='tight')
# %%
print("Minimum at r_H =", xmin)
print("ΔNLL = 1 intersections:", roots)

# %%
from functions import loadMultiParquet_v2, getCommonFilters, loadMultiParquet_Data_new
import pandas as pd
# %%
df = loadMultiParquet_v2(paths=[37], nMCs=-1, filters=getCommonFilters(btagWP="L", cutDijet=False, ttbarCR=False))[0]
df_data, lumi = loadMultiParquet_Data_new(dataTaking=[0], nReals=100, columns=['dijet_mass', 'dijet_pt'], filters=getCommonFilters(btagWP="L", cutDijet=False, ttbarCR=False))
df_data=pd.concat(df_data)
# %%
import uproot
df_data = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/DataPt302025Dec01/ParkingBPH1/crab_data_Run2018A_part1/251201_171548/0000/DataPt30_Run2_data_2025Dec01_210.root")
t = df_data["Events"]
branches = t.arrays(["dijet_mass", "dijet_pt"])
df_data = pd.DataFrame({k: branches[k].to_numpy() for k in ["dijet_mass", "dijet_pt"]})
# %%
signal_eff = df[df.dijet_pt>100].flat_weight.sum()/df.flat_weight.sum()
data_eff = len(df_data[df_data.dijet_pt>100])/len(df_data)
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_pt, bins=np.linspace(30, 200, 51), weights=df.flat_weight, histtype='step', label='ggF H(bb)', density=True)
ax.hist(df_data.dijet_pt, bins=np.linspace(30, 200, 51), histtype='step', label='Data', density=True)
ax.set_xlabel('Dijet $p_{T}$ [GeV]')
ax.set_ylabel('Events [a.u.]')
ax.text(0.95, 0.75, 'jet1 btagWP : L\njet2 btagWP : L', transform=ax.transAxes, fontsize=18,ha='right' )
ax.text(0.95, 0.5, 'Dijet p$_T$ > 100 GeV efficiency\nH(bb) %.1f%%\nData %.1f%%'%(signal_eff*100, data_eff*100), transform=ax.transAxes, fontsize=18,ha='right' )
ax.legend()

# %%
import pandas as pd
df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_pt[df.PNN>0.8], bins=np.linspace(100, 900, 101), density=True, histtype='step', label='H(bb) NN > 0.8')
#ax.hist(df.dijet_pt[df.PNN>0.9], bins=np.linspace(100, 900, 101), density=True, histtype='step', label='PNN > 0.9')
#ax.hist(df.dijet_pt[df.PNN>0.95], bins=np.linspace(100, 900, 101), density=True, histtype='step', label='PNN > 0.95')
ax.legend()
ax.set_xlabel('Dijet $p_{T}$ [GeV]')
# %%
# %%

