import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.pyplot as plt
from ROOT import TCanvas, TGraph, TH1F, TF1, gStyle, TH1D

# Define Python Histogram #
mean, std = 1 , 0.1
data = np.random.normal(mean,std,500)

# Define Bin Size #
xmin = np.floor(10.*data.min())/10.
xmax = np.ceil(10.*data.max())/10.
nbins = int((xmax-xmin)*100)
# print(xmin, xmax, nbins)

# Create Python Histogram #
hist, bin_edges, patches = plt.hist(data,nbins,(xmin,xmax),color='g',alpha=0.6)
bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.

# Find Non-zero bins in Histogram
nz = hist>0
first_nz = bin_centers[nz][ 0] - 0.005
last_nz  = bin_centers[nz][-1] + 0.005


root_hist = np.zeros(nbins+2,dtype=float)
root_hist[1:-1] = hist
h = TH1D('h','hist',nbins,bin_edges)
h.SetContent(root_hist)
h.SetTitle("Root Histogram with Fit;X Axis;Y Axis [Counts]")
c1 = TCanvas("c1", "c1", 800, 600)
# Fit histogram with root #
h.Fit('gaus','','',xmin,xmax)

# Get Root Fit and Goodness of Fit Parameters #
f = h.GetFunction('gaus')
const,mu,sigma = f.GetParameter(0), f.GetParameter(1), f.GetParameter(2)
econst,emu,esigma = f.GetParError(0), f.GetParError(1), f.GetParError(2)
ndf,chi2,prob = f.GetNDF(),f.GetChisquare(),f.GetProb()

print(chi2, ndf)
print(chi2/ndf,prob)
gStyle.SetOptFit(1111)
gStyle.SetOptStat(0000)
# Draw Fit and Histogram#
h.Draw()
c1.Draw()
c1.SaveAs('root_fit.png')