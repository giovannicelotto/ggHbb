# %%
import uproot
import matplotlib.pyplot as plt
f0 = uproot.open("higgsCombinefixed_pdf_0.MultiDimFit.mH125.root")


tree_0 = f0['limit']
branches_0 = tree_0.arrays()
deltaNLL_0 = branches_0["deltaNLL"]
nll_0 = branches_0["nll"]
nll0_0 = branches_0["nll0"]
r_0 = branches_0["r"]

f1 = uproot.open("higgsCombinefixed_pdf_1.MultiDimFit.mH125.root")
tree_1 = f1['limit']
branches_1 = tree_1.arrays()
deltaNLL_1 = branches_1["deltaNLL"]
nll_1 = branches_1["nll"]
nll0_1 = branches_1["nll0"]
r_1 = branches_1["r"]



f_e = uproot.open("higgsCombineEnvelope.MultiDimFit.mH125.root")
tree_e = f_e['limit']
branches_e = tree_e.arrays()
deltaNLL_e = branches_e["deltaNLL"]
nll_e = branches_e["nll"]
nll0_e = branches_e["nll0"]
r_e = branches_e["r"]


fig, ax = plt.subplots(1, 1)
ax.plot(r_0, 2*(deltaNLL_0+nll_0+nll0_0), marker='*', linestyle='none', label='f0')
ax.plot(r_1, 2*(deltaNLL_1+nll_1+nll0_1), marker='*', linestyle='none', label='f1')
ax.plot(r_e, 2*(deltaNLL_e+nll_e+nll0_e), marker='*', linestyle='none', label='Envelope')
ax.legend()
# %%