import subprocess
import uproot, sys
import numpy as np

def get_files_from_das(query):
    # Esegui il comando dasgoclient con la query specificata
    result = subprocess.run(['dasgoclient', '-query=' + query], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Controlla se il comando ha restituito un output corretto
    if result.returncode == 0:
        # Dividi l'output in righe e restituisci la lista dei files
        return result.stdout.strip().split('\n')
    else:
        # Se c'è stato un errore, stampa l'errore e restituisci una lista vuota
        print("Errore durante l'esecuzione del comando:")
        print(result.stderr)
        return []

if __name__ =="__main__":
    query = "file dataset = /GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
    fileNames = get_files_from_das(query)
    print("Lenght of fileNames : %d"%len(fileNames))
    # Stampa la lista dei files ottenuti
    totalMiniEntries = 0

    for fileName in fileNames:
        try:
            with uproot.open("root://cms-xrd-global.cern.ch//" + fileName) as f:
                tree = f['Events']
                maxEntries = tree.num_entries 
                totalMiniEntries += maxEntries
        except:
            print(fileName, " not opened")
        print("%d/%d\n\t\t"%(fileNames.index(fileName)+1, len(fileNames)), totalMiniEntries)
    np.save("/t3home/gcelotto/ggHbb/outputs/counters/N_mini_GluGluHToBB.npy", totalMiniEntries)