import os 

def getInfolderOutfolder():
    outFolder   = "/t3home/gcelotto/ggHbb/PNN/results/basicFeatures" 
    # basicFeatures
    # allFeatures
    # prova
    inFolder    = "/t3home/gcelotto/ggHbb/PNN/input"

    # define suffix = inclusive or other
    if not os.path.exists(inFolder):
        os.makedirs(inFolder)
    if not os.path.exists(outFolder+"/performance"):
        os.makedirs(outFolder+"/performance")
        os.makedirs(outFolder+"/model")

    return inFolder, outFolder
