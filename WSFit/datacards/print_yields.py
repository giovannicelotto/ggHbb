import ROOT, sys

f = ROOT.TFile.Open(sys.argv[1])

for fit in ["shapes_prefit", "shapes_fit_b", "shapes_fit_s"]:
    print(f"\n=== {fit} ===")
    d = f.Get(fit)
    for cat in d.GetListOfKeys():
        catdir = d.Get(cat.GetName())
        print(f"\nCategory: {cat.GetName()}")
        for proc in catdir.GetListOfKeys():
            h = catdir.Get(proc.GetName())
            if isinstance(h, ROOT.TH1):
                print(f"  {proc.GetName():15s} : {h.Integral():.3f}")

