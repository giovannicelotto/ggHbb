# %%
import gzip
from correctionlib import _core
# %%
# Load CorrectionSet
fname = "/t3home/gcelotto/ggHbb/systematics/wpDeepJet/btv-json-sf/data/UL2018/btagging.json.gz"
if fname.endswith(".json.gz"):
    with gzip.open(fname,'rt') as file:
        data = file.read().strip()
        cset = _core.CorrectionSet.from_string(data)
else:
    cset = _core.CorrectionSet.from_file(fname)
# %%
# Step 2: Access the specific correction (deepJet_mujets in this case)
wp_finder = cset["deepJet_wp_values"]
corrDeepJet_shape           = cset["deepJet_shape"]
wp_finder.evaluate("M")
# %%
corr1 = cset["deepJet_mujets"]

# Step 3: Define the parameters for the query
systematic = "central"  
working_point = "L"  
flavor = 5
abseta = 1.0 
pt = 100.0  # Example pt value (GeV)
sf = corrDeepJet_shape.evaluate(systematic, flavor, abseta, pt, 0.9)

print(sf)
# %%
