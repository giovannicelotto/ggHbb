import ROOT
IS_HARD_PROCESS = 128
FROM_HARD_PROCESS = 256
IS_LAST_COPY_BEFORE_FSR = 16384

def has_flag(flags, bit):
    return (flags & bit) != 0

def get_hard_process(pdg, status, flags):
    
    # incoming
    incoming = [
        pdg[i] for i in range(len(pdg))
        if (status[i] == 21 and has_flag(flags[i], IS_HARD_PROCESS))
    ]
    
    # outgoing ME particles
    outgoing = [
        pdg[i] for i in range(len(pdg))
        if (status[i] == 23 and has_flag(flags[i], IS_HARD_PROCESS))
    ]
    
    def label(p):
        if abs(p) in [1,2,3,4,5,6]: return "q"
        if p == 21: return "g"
        return "X"
    
    initial = "".join(sorted(label(p) for p in incoming))
    final   = "".join(sorted(label(p) for p in outgoing))
    
    return initial, final

def get_ME_final(pdg, status):
    return [pdg[i] for i in range(len(pdg)) if status[i] == 23]

def build_children(mothers):
    children = {i: [] for i in range(len(mothers))}
    for i, m in enumerate(mothers):
        if m >= 0:
            children[m].append(i)
    return children

def get_descendants(i, children):
    out = []
    stack = [i]
    
    while stack:
        node = stack.pop()
        for c in children[node]:
            out.append(c)
            stack.append(c)
    
    return out


def produces_bb(i, pdg, children):
    desc = get_descendants(i, children)
    
    b = any(abs(pdg[j]) == 5 for j in desc)
    bbar = any(pdg[j] == -5 for j in desc)
    
    return b and bbar
def produces_cc(i, pdg, children):
    desc = get_descendants(i, children)
    
    c = any(abs(pdg[j]) == 4 for j in desc)
    cbar = any(pdg[j] == -4 for j in desc)
    
    return c and cbar
def classify_leg(i, pdg, children):
    
    p = pdg[i]
    
    # direct b quark
    if abs(p) == 5:
        return "b"
    
    # gluon splitting
    if p == 21:
        if produces_bb(i, pdg, children):
            return "g(bb)"
        elif produces_cc(i, pdg, children):
            return "g(cc)"
        else:
            return "g"
    
    # light quark
    if abs(p) in [1,2,3,4]:
        return "q"
    
    return "X"

def classify_event_ME_based(pdg, status, mothers):
    
    children = build_children(mothers)
    
    # ME final state indices
    me_idx = [i for i in range(len(pdg)) if status[i] == 23]
    
    if len(me_idx) != 2:
        return "weird"
    
    legs = [classify_leg(i, pdg, children) for i in me_idx]
    
    return " + ".join(sorted(legs))
def inv_mass_MEfinalstate(pdg, status, GenPart_pt, GenPart_eta, GenPart_phi):
    
    #children = build_children(mothers)
    
    # ME final state indices
    me_idx = [i for i in range(len(pdg)) if status[i] == 23]
    
    if len(me_idx) != 2:
        return 0.
    else:
        p1 = ROOT.TLorentzVector(0.,0.,0.,0.)
        p2 = ROOT.TLorentzVector(0.,0.,0.,0.)
        p1.SetPtEtaPhiM(GenPart_pt[me_idx[0]], GenPart_eta[me_idx[0]], GenPart_phi[me_idx[0]], 0.139)
        p2.SetPtEtaPhiM(GenPart_pt[me_idx[1]], GenPart_eta[me_idx[1]], GenPart_phi[me_idx[1]], 0.139)
        return (p1+p2).M()

def classify_event_v2(pdg, status, flags, mothers):
    
    initial, final = get_hard_process(pdg, status, flags)
    
    hf_origin = classify_event_ME_based(pdg, status, mothers)

    
    return f"{initial}->{final} | {hf_origin}"