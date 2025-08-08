# %%
import ROOT
x = ROOT.RooRealVar("x", "x", -3,3)
mu = ROOT.RooRealVar("mu", "mu", 0)
sigma = ROOT.RooRealVar("sigma", "sigma", 1, 0.1,9)
sigma.setConstant(True)
gaus = ROOT.RooGaussian("gaus", "gaus", x, mu, sigma)



frame = x.frame()
gaus.plotOn(frame)
canvas = ROOT.TCanvas()
frame.Draw()
canvas.Draw()



# %%
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-3, 3, 100)
y = 1/np.sqrt(2*np.pi)*np.exp(-0.5*(x)**2)
fig, ax = plt.subplots(1, 1)
ax.plot(x, y)
# %%
