# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import uproot
import time
# %%
# To be run after
# 
np.random.seed(1999)
offset=np.random.uniform(10,100)
multiplier=np.random.uniform(10,100)
config = {
    'categories': [0,1,2,3,4,5,6,10, 'Combined'],
    'categoryLabels':[r'NN$_{XXT}$ Btag$_{TT}$',
                      r'NN$_{XT}$ Btag$_{TT}$', 
                      r'NN$_{T}$ Btag$_{TT}$',
                      r'NN$_{M}$ Btag$_{TT}$',
                      r'NN$_{L}$ Btag$_{TT}$',
                      r'NN$_{XL}$ Btag$_{TT}$',
                      r'NN$_{XXL}$ Btag$_{TT}$',
                      r'NN$_{M}$ Btag$_{TT}$',
                      r'NN$_{M}$ Btag$_{MM}$',
                       'Combined'],
    'colors': ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan', 'lightgreen', 
               'lightblue',
               'black'],
    #'nPDFs': 3,
    #'pdfLabels': ['Bern2', 'Expo1', 'PolExpo3'] #0
    #'pdfLabels': ['Bern2', 'Bern3', 'Expo1', 'PolExpo2'] #10
    #'pdfLabels': ['Bern2', 'Expo1', 'PolExpo2'] #li
}

# %%
r_Discrete_cat_i = []
deltaNLL_Discrete_cat_i = []
nll0_cat_i = []
nll_cat_i = []
for cat in config['categories']:
    if cat == 'Combined':
        continue
    fileName = "/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinefitData_Blind_Cat%dpdfDiscrete.MultiDimFit.mH125.root"%cat
    file = uproot.open(fileName)
    tree = file["limit"]
    branches = tree.arrays(library="np")

    r_Discrete_cat_i.append((branches["r"]+offset) * multiplier)
    deltaNLL_Discrete_cat_i.append(branches["deltaNLL"])
    #nll0_cat_i.append(branches["nll0"])     not useful here
    #nll_cat_i.append(branches["nll"])   not useful here
    deltaNLL_Discrete_cat_i[-1] -= min(deltaNLL_Discrete_cat_i[-1])

fileName = "/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinefitData_Blind_CatCombined_pdfDiscrete.MultiDimFit.mH125.root"
file = uproot.open(fileName)
tree = file["limit"]
branches = tree.arrays(library="np")

r_Discrete_cat_i.append((branches["r"]+offset) * multiplier)
deltaNLL_Discrete_cat_i.append(branches["deltaNLL"])
    



# %%
fig, ax = plt.subplots(1,1 )

shift_for_visual = 0
xmin = 8000
xmax = -8000
for i,cat in enumerate(config['categories']):
    print(f"Category {cat}:")
    ax.plot(r_Discrete_cat_i[i][1:],deltaNLL_Discrete_cat_i[i][1:], 'o', markersize=3, linestyle='-', linewidth=1, alpha=0.3, color=config['colors'][i])
    
    
    y_Discrete_left = deltaNLL_Discrete_cat_i[i][r_Discrete_cat_i[i] < r_Discrete_cat_i[i][np.argmin(deltaNLL_Discrete_cat_i[i])]]
    r_Discrete_left = r_Discrete_cat_i[i][r_Discrete_cat_i[i] < r_Discrete_cat_i[i][np.argmin(deltaNLL_Discrete_cat_i[i])]]
    y_Discrete_right = deltaNLL_Discrete_cat_i[i][np.argmin(deltaNLL_Discrete_cat_i[i][1:]):]
    r_Discrete_right = r_Discrete_cat_i[i][np.argmin(deltaNLL_Discrete_cat_i[i][1:]):]
    x_intersectionLeft2Sigma, x_intersectionRight2Sigma = r_Discrete_left[np.argmin(np.abs(y_Discrete_left - (min(y_Discrete_left)+4)))], r_Discrete_right[np.argmin(np.abs(y_Discrete_right - (min(y_Discrete_right)+4)))]
    x_intersectionLeft1Sigma, x_intersectionRight1Sigma = r_Discrete_left[np.argmin(np.abs(y_Discrete_left - (min(y_Discrete_left)+1)))], r_Discrete_right[np.argmin(np.abs(y_Discrete_right - (min(y_Discrete_right)+1)))]
    rMin = r_Discrete_cat_i[i][np.argmin(deltaNLL_Discrete_cat_i[i])]
    
    ax.errorbar(
        x=rMin, y=shift_for_visual+min(deltaNLL_Discrete_cat_i[i])-0.25,
        xerr=np.array([[rMin - x_intersectionLeft2Sigma],     [x_intersectionRight2Sigma - rMin]     ]),
        yerr=0,fmt='o',color=config['colors'][i],markersize=10,linewidth=4,alpha=0.2,
    )
    ax.errorbar(
        x=rMin,
        y=shift_for_visual+min(deltaNLL_Discrete_cat_i[i])-0.25,
        xerr=np.array([[rMin - x_intersectionLeft1Sigma],     [x_intersectionRight1Sigma - rMin]     ]),
        yerr=0,fmt='o',color=config['colors'][i],markersize=10,linewidth=2,label=f'Category {config["categoryLabels"][i]}' + " (r = %.2f$_{%.2f}^{+%.2f}$)"%(rMin, x_intersectionLeft1Sigma-rMin, x_intersectionRight1Sigma-rMin)
    )
    #ax.text(ax.get_xlim()[1]*1.05, shift_for_visual+min(deltaNLL_Discrete_cat_i[i])-0.1,
    #        "r = %.2f$_{%.2f}^{+%.2f}$ "%(rMin, x_intersectionLeft1Sigma-rMin, x_intersectionRight1Sigma-rMin), fontsize=14, ha='left', va='top')

    if  x_intersectionLeft2Sigma < xmin:
        xmin =  x_intersectionLeft2Sigma 
    if  x_intersectionRight2Sigma > xmax:
        xmax =  x_intersectionRight2Sigma 
    ax.set_xlim(xmin, xmax)

    shift_for_visual-=.25

ax.hlines(1, xmin=min(r_Discrete_cat_i[-2]), xmax=max(r_Discrete_cat_i[-2]), colors='black',alpha=0.5, linestyles='dashed')
ax.hlines(4, xmin=min(r_Discrete_cat_i[-2]), xmax=max(r_Discrete_cat_i[-2]), colors='black',alpha=0.2, linestyles='dashed')
ax.text(0.85, 0.95, "r = %.2f$_{%.2f}^{+%.2f}$ "%(rMin, x_intersectionLeft1Sigma-rMin, x_intersectionRight1Sigma-rMin), transform=ax.transAxes, fontsize=18, ha='right', va='top')
ax.set_xlabel("(r + RND$_1$) x RND$_2$")
ax.set_ylabel("-2 $\Delta$ ln(L)")
ax.set_ylim(-3.5, 6)
#ax.set_xlim(min(r_Discrete_cat_i[-2]), max(r_Discrete_cat_i[-2]))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plotName=f"/t3home/gcelotto/ggHbb/WSFit/output/combined/DiscreteNLLScan_AllCategories_RNDShifted.png"
fig.savefig(plotName, bbox_inches='tight')
print("Saved plot to ", plotName)
# %%
