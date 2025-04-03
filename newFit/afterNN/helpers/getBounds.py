import yaml
def getBounds(paramFile = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/config_1.yaml"):
    with open(paramFile, "r") as f:
        params = yaml.safe_load(f)  # Read as dictionary

    x1 = params["x1"]
    x2 = params["x2"]
    key = params["key"]
    nbins = params["nbins"]
    t1 = params["t1"]
    t2 = params["t2"]
    t0 = params["t0"]
    t3 = params["t3"]
    return x1, x2, key, nbins, t1, t2, t0, t3