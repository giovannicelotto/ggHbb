# %%
import gzip
import json
# %%
# Path al file
path = "/t3home/gcelotto/ggHbb/LeptonSF/electron.json.gz"

# Apro il file e carico il contenuto
with gzip.open(path, "rt") as f:
    data = json.load(f)

# Stampare tutto il contenuto (per ispezione)
print(data['key'])

# Oppure scorrere chiavi e valori se è un dict
if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {value}")

# Oppure scorrere elementi se è una lista
elif isinstance(data, list):
    for elem in data:
        print(elem)

# %%
