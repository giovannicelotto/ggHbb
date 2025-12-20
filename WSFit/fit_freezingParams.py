# %%
import subprocess
import re, sys
import numpy as np
# Build the shell command as a single string
def findFamilyAndOrder(mystring, verbose=False):
    if "Bernstein" in mystring:
        type = "Bernstein"
    elif "Exponential" in mystring:
        type = "Exponential"
    if "PowerLaw" in mystring:
        type = "PowerLaw"
    order = -1
    for i in range(7):
        if type + "_%d"%i in mystring:
            order = i
    if verbose:
        print(f"Matched : {type} {order}")
    return type, order

catIdx = 2

# get the ws

cmd = f"""
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
cmsenv &&
root -l -b -q -e 'TFile* f = new TFile("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{catIdx}.root");
RooWorkspace* ws3 = (RooWorkspace*) f->Get("ws3");
RooMultiPdf* mpdf = (RooMultiPdf*) ws3->pdf("CMS_hgg_0_2016_13TeV_bkgshape_noZ");
mpdf->getCurrentPdf()->Print();
'
"""



    # Run in a bash shell
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
bestFamily, bestOrder = findFamilyAndOrder(result.stdout)


cmd = f"""
cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src &&
cmsenv &&
root -l -b -q -e 'TFile* f = new TFile("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{catIdx}.root");
RooWorkspace* ws3 = (RooWorkspace*) f->Get("ws3");
RooMultiPdf* mpdf = (RooMultiPdf*) ws3->pdf("CMS_hgg_0_2016_13TeV_bkgshape_noZ");
RooCategory* cat = (RooCategory*) ws3->cat("pdfindex_{catIdx}_2016_13TeV");
int numTypes = cat->numTypes();
std::cout << "Number of PDFs: " << numTypes << std::endl; \
for (int i = 0; i < numTypes; ++i) {{ \
    const RooAbsPdf* pdf = mpdf->getPdf(i); \
    if (pdf) std::cout << "PDF " << i << ": " << pdf->GetName() << std::endl; \
}}
'
"""



    # Run in a bash shell
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
families, orders = [], []
for line in result.stdout.splitlines():
    if "env_pdf" in line:
        family, order = findFamilyAndOrder(line)
        families.append(family)
        orders.append(order)
print(families)        
print(orders)


# print all the paramters

sys.exit()

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
combineCards.py datacardMulti2.txt datacardMulti4.txt datacardMulti14.txt datacardMulti5.txt datacardMulti15.txt  datacardMulti1.txt datacardMulti0.txt datacardMulti12.txt datacardMulti11.txt datacardMulti10.txt  > combinedDataCard.txt
combine -M Significance -d combinedDataCard.txt \
    --freezeParameters pdfindex_2_2016_13TeV,pdfindex_1_2016_13TeV,pdfindex_0_2016_13TeV,pdfindex_12_2016_13TeV,pdfindex_11_2016_13TeV,pdfindex_10_2016_13TeV \
    -t -1 \
    --expectSignal 1
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
    #print(resultBkg.stdout)
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
print(signalYieldsMatrix)
print(dataYieldsMatrix)
print(significanceMatrix)
print(signalYieldsMatrix/np.sqrt(dataYieldsMatrix))
print(np.sqrt(np.sum((signalYieldsMatrix/np.sqrt(dataYieldsMatrix))**2)))


# %%
