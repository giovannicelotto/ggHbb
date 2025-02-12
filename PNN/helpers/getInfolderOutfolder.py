import os 

def getInfolderOutfolder(name, suffixResults="", createFolder=True):
    outFolder   = "/t3home/gcelotto/ggHbb/PNN/results%s/"%suffixResults+name
    # basicFeatures
    # allFeatures
    # prova
    inFolder    = "/t3home/gcelotto/ggHbb/PNN/input"

    # define suffix = inclusive or other
    if createFolder:
        if not os.path.exists(inFolder):
            os.makedirs(inFolder)
        if not os.path.exists(outFolder+"/performance"):
            os.makedirs(outFolder+"/performance")
            os.makedirs(outFolder+"/model")
    
    print("InFolder :", inFolder)
    print("OutFolder :", outFolder)
    return inFolder, outFolder
