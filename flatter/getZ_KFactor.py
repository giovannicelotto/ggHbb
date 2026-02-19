# %%
import yaml
import numpy as np
def getZ_KFactor(pt, path="/t3home/gcelotto/ggHbb/Z_kfactor/output/kfactor.json"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    bins = config["bins"]
    kfactor = config["kfactor"]
    bin_idx = np.digitize(pt, bins=bins)
    if bin_idx == 0:
        #underflow
        pass
    elif pt>=bins[-1]:
        #overflow
        bin_idx = bin_idx -2
    else:
        bin_idx = bin_idx-1
    #overflow not possible.
    assert len(bin_idx)==1
    try:
        result = kfactor[bin_idx[0]]
    except:
        print("pt=", pt, "bin_idx=", bin_idx, "bins=", bins, "kfactor=", kfactor)
    return result
#if __name__=="__main__":
#    getZ_KFactor(np.array([1.]), path="/t3home/gcelotto/ggHbb/Z_kfactor/output/kfactor.json")
# %%
