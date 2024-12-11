# %%
from correctionlib import _core
import gzip

# %%
fname = "/t3home/gcelotto/ggHbb/systematics/wpDeepJet/btv-json-sf/data/UL2018/btagging.json.gz"
if fname.endswith(".json.gz"):

    with gzip.open(fname,'rt') as file:
        #data = json.load(file)
        data = file.read().strip()
        cset = _core.CorrectionSet.from_string(data)
else:
    cset = _core.CorrectionSet.from_file(fname)
# %%
corrFixedWP_muJets = cset["deepJet_mujets"]
btag_systs = ['central','down','down_jes', 'down_pileup', 'down_statistic', 'down_type3', 'up', 'up_jes', 'up_pileup', 'up_statistic', 'up_type3', 'down_correlated', 'down_uncorrelated', 'up_correlated', 'up_uncorrelated']
wp_converter = cset["deepJet_wp_values"]
# %%
