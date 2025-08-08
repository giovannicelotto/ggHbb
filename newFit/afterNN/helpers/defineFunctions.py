from helpers.allFunctions import *

def defineFunctions(key):
    # Centralized list for better readability and consistency
    entries = [
        # key | bkg function                        | bkg+signal function                   | parameters
        (0,    f0_mod,                    f0_mod_DSCB,                  ["normBkg", "xo", "k", "delta", "beta"]),
        (1,    f1_mod,                    f1_mod_DSCB,                  ["normBkg", "B", "b", "c", "C", "f"]),
        (14,     exp_pol2_turnOn,                 exp_pol2_turnOn_DSCB,               ["normBkg", "B", "b", "c", "aa", "bb", "f"]),
        (15,     sum_exp,                 sum_exp_DSCB,               ["normBkg", "x0", "A1", "tau1", "A2", "tau2", "tau3"]),
        (16,     sigmoidBased,                 sigmoidBased_DSCB,               ["normBkg", "p1_fall", "p0_rise", "p1_rise", "p2_rise", "x0", "k"]),


        (7,     expExp_pol2_turnOn,                 expExp_pol2_turnOn_DSCB,               ["normBkg", "B", "C", "b", "c", "aa", "bb", "f"]),
        (11,    expPol2_expPol2,                    expPol2_expPol2_DSCB,                  ["normBkg", "B", "C", "p1", "p2", "p11", "p12", "f"]),
        (20,    f0,                    f0_DSCB,                  ["normBkg", "xo", "k", "beta"]),
        (21,    gamma,                    gamma_DSCB,                  ["normBkg", "xo","a", "scale"]),
        

        #(0,     expExpExp,                          expExpExp_DSCB,                         ["normBkg", "B", "C", "D"]),
        #(1,     exp_pol1,                           continuum_DSCB1,                        ["normBkg", "B", "b"]),
        (2,     exp_pol2,                           continuum_DSCB2,                        ["normBkg", "B", "b", "c"]),
        (3,     exp_pol3,                           continuum_DSCB3,                        ["normBkg", "B", "b", "c", "d"]),
        (5,     exp_pol5,                           continuum_DSCB5,                        ["normBkg", "B", "b", "c", "d", "e", "f"]),
        (6,     expExp_pol2,                        expExpPol2_DSCB,                        ["normBkg", "B", "C", "b", "c"]),       
        (8,     exp_pol2_turnOn3,                   exp_pol2_turnOn3_DSCB,                 ["normBkg", "B", "C", "b", "c", "aa", "bb", "cc", "f"]),
        (9,     exp_gaus_turnOn,                    exp_gaus_turnOn_DSCB,                  ["normBkg", "B", "b", "c", "mu_to", "sigma_to", "f"]),
        (10,    expExp_pol2_turnOnPol3,             expExp_pol2_turnOnPol3_DSCB,           ["normBkg", "B", "C", "b", "c", "d", "aa", "bb", "f"]),
        (12,    expExp_pol2_turnOnPol1,             expExp_pol2_turnOnPol1_DSCB,           ["normBkg", "B", "C", "b", "c", "aa", "f"]),
        (13,    pol3,                               pol3_DSCB,                              ["normBkg", "p1", "p2", "p3"]     )

        # RSCB versions
        #(101,  exp_pol1,                             continuum_RSCB1,                        ["normBkg", "B", "b"]),
        #(102,  exp_pol2,                             continuum_RSCB2,                        ["normBkg", "B", "b", "c"]),
        #(103,  exp_pol3,                             continuum_RSCB3,                        ["normBkg", "B", "b", "c", "d"]),
        #(106,  expExp_pol2,                          expExpPol2_RSCB,                        ["normBkg", "B", "C", "b", "c"]),
        #(107,  expExp_pol2_turnOn,                   expExp_pol2_turnOn_RSCB,               ["normBkg", "B", "C", "b", "c", "aa", "bb", "f"]),
        #(108,  exp_pol2_turnOn3,                     exp_pol2_turnOn3_RSCB,                 ["normBkg", "B", "b", "c", "aa", "bb", "cc", "f"]),
        #(109,  exp_gaus_turnOn,                      exp_gaus_turnOn_RSCB,                  ["normBkg", "B", "b", "c", "mu_to", "sigma_to", "f"]),
        #(110,  kinThreshold,                         kinThreshold_RSCB,                     ["normBkg", "B", "m0", "p1", "p2"]),
    ]

    myBkgFunctions = {k: bkg for k, bkg, _, _ in entries}
    myBkgSignalFunctions = {k: bkgSig for k, _, bkgSig, _ in entries}
    myBkgParams = {k: params for k, _, _, params in entries}
    if key==11:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], expPol2_expPol2_DSCB_Z_H
    elif key==14:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], exp_pol2_turnOn_DSCB_Z_H
    elif key==0:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], f0_mod_DSCB_Z_H
    elif key==1:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], f1_mod_DSCB_Z_H
    elif key==15:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], sum_exp_DSCB_Z_H
    elif key==16:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], sigmoidBased_DSCB_Z_H
    elif key==7:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], expExp_pol2_turnOn_DSCB_Z_H
    elif key==20:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], f0_DSCB_Z_H
    elif key==21:
        return myBkgFunctions[key], myBkgSignalFunctions[key], myBkgParams[key], gamma_DSCB_Z_H
    else:
        assert False, "not implemented yet"
