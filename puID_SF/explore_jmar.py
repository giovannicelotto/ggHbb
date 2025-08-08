from correctionlib import _core

# Load CorrectionSet
fname = "/t3home/gcelotto/ggHbb/puID_SF/jmar.json.gz"
if fname.endswith(".json.gz"):
    import gzip
    with gzip.open(fname,'rt') as file:
        data = file.read().strip()
        evaluator = _core.CorrectionSet.from_string(data)
else:
    evaluator = _core.CorrectionSet.from_file(fname)


eta, pt, syst, wp = 2.0,20.,"up","L"
map_name = "PUJetID_eff"
valsf= evaluator[map_name].evaluate(eta, pt, syst, wp)
print("Example for "+map_name)
print("The "+syst+" SF for a Jet with pt="+str(pt) + " GeV and eta="+str(eta) + " for the "+wp+" working point is "+str(valsf))
