import numpy as np
import sys
import glob
import os
import re
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatData"
lightFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/lightFlatData"
fileNames = glob.glob(folder+"/.npy")
file_list = os.listdir(folder)
npy_files = [file for file in file_list if file.endswith('.npy')]
for fileName in fileNames:
    #path = folder + "/ggHbb_bScoreBased4_171.npy"
    file = np.load(fileName)
    fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
#print(type(file[0,0]))
#dtypes = []
#for i in range(len(getTypes())):
#    dtypes.append((getFeaturesBScoreBased()[i],getTypes()[i]))
#
#print(file.shape[1], len(dtypes), len(getFeaturesBScoreBased()), len(getTypes()))
    
    file = np.array(file, dtype=np.float32)
#print(np.array(dtypes)[:,1])
    np.save(lightFolder+"/ggHbb_bscoreBased4_%d"%fileNumber, file)