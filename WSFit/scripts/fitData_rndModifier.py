# %%
import subprocess
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import mplhep as hep


category = 10
pdfLabels = ["Bern2", "Expo1", "PolExpo2"]
pdfIdxs = np.arange(len(pdfLabels))
# %%
np.random.seed(int(time.time()))
offset=np.random.uniform(10,100)
multiplier=np.random.uniform(10,100)
# %%

dataset = {
    'category' : [],
    'r' : [],
    'rLoErr' : [],
    'rHiErr' : [],
}
# %%
for pdfIdx in pdfIdxs:
    
    cmd = f"""
    cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
    cmsenv &&
    cd /t3home/gcelotto/ggHbb/WSFit/datacards &&
        combine -M FitDiagnostics -d datacardMulti{category}.txt  \
        --setParameterRanges r=-20,20  -m 125  \
        --setParameters pdfindex_{category}_2016_13TeV={pdfIdx}  \
        --freezeParameters pdfindex_{category}_2016_13TeV\
        --cminDefaultMinimizerStrategy=0\
        --X-rtd MINIMIZER_freezeDisassociatedParams  --rMin -7 --rMax 9 \
        -n fitData_Blind_pdf{pdfIdx}
    """

    # Run in a bash shell
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


    # Extract the significance number
    pattern = r"Best fit r:\s*([+-]?\d*\.?\d+)\s*([+-]?\d*\.?\d+)/\+?([+-]?\d*\.?\d+)"

    m = re.search(pattern, result.stdout)
    if m:
        r_val     = float(m.group(1))
        low_err   = float(m.group(2))
        high_err  = float(m.group(3))

        #print((r_val+offset)*multiplier, multiplier*(low_err+r_val+offset), multiplier*(high_err+r_val+offset))
        dataset['r'].append((r_val+offset)*multiplier)
        dataset['category'].append(category)
        dataset['rLoErr'].append(abs(low_err)*multiplier)
        dataset['rHiErr'].append(high_err*multiplier)
    
    else:
        print("Could not find r value in the output")



# %%

hep.style.use("CMS")
fig, ax = plt.subplots(1,1)
ax.errorbar(np.arange(len(dataset['r'])), dataset['r'], yerr=[dataset['rLoErr'], dataset['rHiErr']], fmt='o')
ax.set_xlabel('PDF Index')
ax.set_xticks(np.arange(len(dataset['r'])))
ax.set_xticklabels(pdfLabels)
ax.set_ylabel('(Fitted r + rnd offset) x rnd multiplier')
# %%
# Draw a matrix 3x3 and plot with color bar the compatibility r0-r1/sqrt(sigma0^2+sigma1^2)
fig, ax = plt.subplots(len(pdfIdxs), len(pdfIdxs), figsize=(10,10))

plt.subplots_adjust(wspace=0, hspace=-0.42)
for i, pdfIdx1 in enumerate(pdfIdxs):
    for j, pdfIdx2 in enumerate(pdfIdxs):
        if i>j:
            ax[i,j].axis('off')
            continue
        r1 = dataset['r'][i]
        r2 = dataset['r'][j]
        sigma1 = 0.5*(dataset['rHiErr'][i] + dataset['rLoErr'][i])
        sigma2 = 0.5*(dataset['rHiErr'][j] + dataset['rLoErr'][j])
        compatibility = abs(r1 - r2) / np.sqrt(sigma1**2 + sigma2**2)
        deltaR = abs(r1 - r2) 
        im = ax[i,j].imshow([[compatibility]], vmin=0, vmax=5, cmap='RdYlGn_r', alpha=0.4)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        # 
        # 
        ax[i,j].text(x=0.5,y=0.5, ha='center',va='center', s='%s vs %s\n $\\frac{|r_1-r_2|}{\sqrt{\sigma_1^2+\sigma_2^2}} =  %.2f$'%(pdfLabels[i], pdfLabels[j], compatibility), transform=ax[i,j].transAxes, color='white' if compatibility>2.5 else 'black', fontsize=14)
        ax[i,j].text(x=0.5,y=0.25, ha='center',va='center', s='$\\Delta r =  %.1f$'%deltaR, transform=ax[i,j].transAxes,  fontsize=12)

# Add colorbar
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8)
# %%
