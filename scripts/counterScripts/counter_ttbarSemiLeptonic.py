import subprocess
import uproot
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

# Esempio di utilizzo
query = "file dataset = /TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM"
fileNames = get_files_from_das(query)

# Stampa la lista dei files ottenuti
totalMiniEntries = 0

    
    
for fileName in fileNames:
    with uproot.open("root://cms-xrd-global.cern.ch//" + fileName) as f:
        tree = f['Events']
        maxEntries = tree.num_entries 
        totalMiniEntries += maxEntries
    print("%d/%d\n\t\t"%(fileNames.index(fileName)+1, len(fileNames)), totalMiniEntries)
np.save("/t3home/gcelotto/ggHbb/outputs/counters/N_mini_ttbarSemiLeptonic.npy", totalMiniEntries)