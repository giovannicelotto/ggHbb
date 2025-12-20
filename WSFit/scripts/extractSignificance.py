# %%
import subprocess
import re
import numpy as np
# Build the shell command as a single string
categories = [0,10,100,1,11,12]
assert len(categories)%2==0
significanceIdx = np.array([[0,10,100],[1,11,101]])
significanceMatrix = np.zeros((2,len(categories)//2))
# %%
for catIdx in categories:
    cmd = f"""
    cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
    cmsenv &&
    cd /t3home/gcelotto/ggHbb/WSFit/datacards &&
    combine -M Significance -d datacardMulti{catIdx}.txt \
        -t -1 --expectSignal 1 --X-rtd MINIMIZER_freezeDisassociatedParams --setParameterRange r=-5,7 \
    """

    # Run in a bash shell
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Print all output
    print(result.stdout)

    # Extract the significance number
    match = re.search(r"Significance:\s+([0-9.]+)", result.stdout)
    if match:
        significance = float(match.group(1))
        print("Extracted significance:", significance)
    else:
        print("Could not find significance in output")
        significance = 0


    significanceMatrix = np.where(significanceIdx==catIdx, significance, significanceMatrix)
    significanceMatrix

    print(significanceMatrix)

# %%
#    
#       Combined Value
#
#
cmd = f"""

cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
cmsenv &&
cd /t3home/gcelotto/ggHbb/WSFit/datacards &&
rm combinedDataCard.txt
combineCards.py datacardMulti0.txt datacardMulti1.txt datacardMulti10.txt datacardMulti11.txt  > combinedDataCard.txt
combine -M Significance -d combinedDataCard.txt \
    --X-rtd MINIMIZER_freezeDisassociatedParams \
    --setParameterRange r=-5,7 \
    -t -1 \
    --expectSignal 1 \
"""

# Run in a bash shell
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Print all output
print(result.stdout)

# Extract the significance number
match = re.search(r"Significance:\s+([0-9.]+)", result.stdout)
if match:
    significance = float(match.group(1))
    print("Extracted significance:", significance)
else:
    print("Could not find significance in output")


# %%
valuesSignal = {}
valuesData = {}
for catIdx in categories:
    cmd = f"""
    cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
    cmsenv &&
    root -l -b -q -e 'TFile* f = new TFile("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{catIdx}.root"); 
                      ws3->cd();
                      model_H_c{catIdx}_norm->Print();'
    """
    resultSignal = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cmdBkg = f"""
    cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
    cmsenv &&
    root -l -b -q -e 'TFile* f = new TFile("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{catIdx}.root");
    RooWorkspace* ws3 = (RooWorkspace*) f->Get("ws3");
    RooAbsPdf* pdf = (RooAbsPdf*) ws3->pdf("CMS_hgg_0_2016_13TeV_bkgshape");
    RooRealVar* dijet_mass = (RooRealVar*) ws3->var("dijet_mass_c{catIdx}");
    dijet_mass->setRange("SR", 100, 150);
    RooArgSet obs(*dijet_mass);
    pdf->createIntegral(obs, RooFit::NormSet(obs), RooFit::Range("SR"))->Print();
    RooRealVar* CMS_hgg_0_2016_13TeV_bkgshape_norm = (RooRealVar*) ws3->var("CMS_hgg_0_2016_13TeV_bkgshape_norm");
    CMS_hgg_0_2016_13TeV_bkgshape_norm->Print();
    '
    """
    #
#    
#
    resultBkg = subprocess.run(cmdBkg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(resultBkg.stdout)
    match = re.search(r"=\s*([0-9.+-eE]+)", resultSignal.stdout)
    frac_match = re.search(r'^\s*([0-9]*\.?[0-9]+)\s*$', resultBkg.stdout, re.MULTILINE)
    fraction = float(frac_match.group(1)) if frac_match else None
    print("Fraction is ", fraction)

    norm_match = re.search(r'RooRealVar::\S+_norm\s*=\s*([0-9.eE+-]+)', resultBkg.stdout)
    if norm_match:
        norm = float(norm_match.group(1))
        print(norm)
    print(resultBkg.stdout)
    norm = float(norm_match.group(1)) if norm_match else None
    print("norm is ", norm)
    valuesData[catIdx] = norm*fraction

    if match:
        val = float(match.group(1))
        valuesSignal[catIdx] = val
        print(f"Signal Yields = {val}")
    else:
        print(f"catIdx={catIdx}, could not parse value")

# %%
signalYieldsMatrix = np.zeros((2,len(categories)//2))
dataYieldsMatrix = np.zeros((2,len(categories)//2))

for i in range(significanceIdx.shape[0]):
    for j in range(significanceIdx.shape[1]):
        idx = significanceIdx[i,j]
        signalYieldsMatrix[i,j] = valuesSignal[idx]
        dataYieldsMatrix[i,j] = valuesData[idx]
# %%
def print_matrix(title, matrix, digits=3):
    fmt = f"{{val:10.{digits}f}}"
    print(f"\n{title}")
    print("-" * (len(matrix[0]) * (digits + 20)))  # adjust line width dynamically
    
    for row in matrix:
        formatted_row = " | ".join(fmt.format(val=val) for val in row)
        print(f"| {formatted_row} |")
    
    print("-" * (len(matrix[0]) * (digits + 20)))
# Derived values
s_over_sqrtb = signalYieldsMatrix / np.sqrt(dataYieldsMatrix)
s_over_b = signalYieldsMatrix / dataYieldsMatrix

# Print neatly
print_matrix("Signal Yields :", signalYieldsMatrix, digits=1)
print_matrix("Background Yields :", dataYieldsMatrix, digits=0)
print_matrix("Significance :", significanceMatrix, digits=3)
print_matrix("S/sqrt(B) :", s_over_sqrtb, digits=5)
print_matrix("S/B :", s_over_b, digits=5)
#print(np.sqrt(np.sum((signalYieldsMatrix/np.sqrt(dataYieldsMatrix))**2)))
# %%
