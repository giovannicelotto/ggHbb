# %%
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
# Open file
file = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ_noTrig2025Jun03/ZJetsToQQ_HT-100to200/ZJetsToQQ_HT100to200_Run2_mc_124X_999.root")


# Open ROOT file
tree = file["Events"]

# Load only the necessary LHE branches
branches = tree.arrays([
    "LHEPart_pt", "LHEPart_eta", "LHEPart_phi", "LHEPart_mass",
    "LHEPart_status", "LHEPart_pdgId"
], library="ak")

# Final-state partons (status == 1 and pdgId is 1–5 or 21)
is_final = branches["LHEPart_status"] == 1
is_parton = (abs(branches["LHEPart_pdgId"]) <= 5) | (abs(branches["LHEPart_pdgId"]) == 21)
mask = is_final & is_parton

# Apply mask
pt = branches["LHEPart_pt"][mask]
eta = branches["LHEPart_eta"][mask]
phi = branches["LHEPart_phi"][mask]
mass = branches["LHEPart_mass"][mask]

# Keep events with exactly two partons
mask_2partons = ak.num(pt) == 2
pt = pt[mask_2partons]
eta = eta[mask_2partons]
phi = phi[mask_2partons]
mass = mass[mask_2partons]

#pt = pt[:,:2]
#eta = eta[:,:2]
#phi = phi[:,:2]
#mass = mass[:,:2]

# Compute px, py, pz, E
px = pt * np.cos(phi)
py = pt * np.sin(phi)
pz = pt * np.sinh(eta)
E  = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

# Sum over both partons in each event
px_sum = ak.sum(px, axis=1)
py_sum = ak.sum(py, axis=1)
pz_sum = ak.sum(pz, axis=1)
E_sum  = ak.sum(E,  axis=1)

# Invariant mass
inv_mass = np.sqrt(E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2)

# Plot
plt.hist(inv_mass, bins=np.linspace(0, 200, 101), histtype="step")
plt.xlabel("Invariant Mass [GeV]")
plt.ylabel("Events")
plt.title("Invariant Mass of 2 LHE Hardest Partons (Npartons==2)")
plt.grid(True)
plt.show()

# %%
