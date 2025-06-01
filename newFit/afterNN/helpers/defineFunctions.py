from helpers.allFunctions import *
def defineFunctions():
    myBkgFunctions = {
    0:expExpExp,
    1:exp_pol1,
    2:exp_pol2,
    3:exp_pol3,
    5:exp_pol5,
    6:expExp_pol2,
    7:expExp_pol2_turnOn,
    8:exp_pol2_turnOn3,
    9:exp_gaus_turnOn,
    10:expExp_pol2_turnOnPol3,
    11:expPol2_expPol2,
    12:expExp_pol2_turnOnPol1,
# For RSCB?
    101:exp_pol1,
    102:exp_pol2,
    103:exp_pol3,
    106:expExp_pol2,
    107:expExp_pol2_turnOn,
    108:exp_pol2_turnOn3,
    109:exp_gaus_turnOn,
    110:kinThreshold


    }

    myBkgSignalFunctions = {
    0:expExpExp_DSCB,
    1:continuum_DSCB1,
    2:continuum_DSCB2,
    3:continuum_DSCB3,
    5:continuum_DSCB5,
    6:expExpPol2_DSCB,
    7:expExp_pol2_turnOn_DSCB,
    8:exp_pol2_turnOn3_DSCB,
    9:exp_gaus_turnOn_DSCB,
    10:expExp_pol2_turnOnPol3_DSCB,
    11:expPol2_expPol2_DSCB,
    12:expExp_pol2_turnOnPol1_DSCB,

    101:continuum_RSCB1,
    102:continuum_RSCB2,
    103:continuum_RSCB3,
    106:expExpPol2_RSCB,
    107:expExp_pol2_turnOn_RSCB,
    108:exp_pol2_turnOn3_RSCB,
    109:exp_gaus_turnOn_RSCB,
    110:kinThreshold_RSCB
    }

    myBkgParams = {
    0:["normBkg", "B", "C", "D"],
    1:["normBkg", "B", "b"],
    2:["normBkg", "B", "b", "c"],
    3:["normBkg", "B", "b", "c", "d"],
    5:["normBkg", "B", "b", "c", "d", "e", "f"],
    6:["normBkg", "B", "C", "b", "c"],
    7:["normBkg", "B", "C", "b", "c", "aa", "bb", "f"],
    8:["normBkg", "B", "C", "b", "c", "aa", "bb", "cc","f"],
    9:["normBkg", "B", "b", "c", "mu_to", "sigma_to", "f"],
    10:["normBkg", "B", "C", "b", "c", "d","aa", "bb", "f"],
    11:["normBkg", "B", "C", "p1", "p2", "p11","p12", "f"],
    12:["normBkg", "B", "C", "b", "c", "aa", "f"],
    
    101:["normBkg", "B", "b"],
    102:["normBkg", "B", "b", "c"],
    103:["normBkg", "B", "b", "c", "d"],
    106:["normBkg", "B", "C", "b", "c"],
    107:["normBkg", "B", "C", "b", "c", "aa", "bb", "f"],
    108:["normBkg", "B", "b", "c", "aa", "bb", "cc","f"],
    109:["normBkg", "B", "b", "c", "mu_to", "sigma_to", "f"],
    110:["normBkg", "B", "m0", "p1", "p2"]
}
    return myBkgFunctions, myBkgSignalFunctions, myBkgParams
