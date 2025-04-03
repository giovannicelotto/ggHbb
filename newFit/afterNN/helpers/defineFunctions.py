from helpers.allFunctions import *
def defineFunctions():
    myBkgFunctions = {
    0:expExpExp,
    1:exp_pol1,
    2:exp_pol2,
    3:exp_pol3,
    5:exp_pol5,
    6:expExp_pol2,
# For RSCB?
    101:exp_pol1,
    102:exp_pol2,
    103:exp_pol3,
    106:expExp_pol2,


    }

    myBkgSignalFunctions = {
    0:expExpExp_DSCB,
    1:continuum_DSCB1,
    2:continuum_DSCB2,
    3:continuum_DSCB3,
    5:continuum_DSCB5,
    6:expExpPol2_DSCB,

    101:continuum_DSCB1,
    102:continuum_DSCB2,
    103:continuum_DSCB3,
    106:expExpPol2_RSCB,
    }

    myBkgParams = {
    0:["normBkg", "B", "C", "D"],
    1:["normBkg", "B", "b"],
    2:["normBkg", "B", "b", "c"],
    3:["normBkg", "B", "b", "c", "d"],
    5:["normBkg", "B", "b", "c", "d", "e", "f"],
    6:["normBkg", "B", "C", "b", "c"],
    
    101:["normBkg", "B", "b"],
    102:["normBkg", "B", "b", "c"],
    103:["normBkg", "B", "b", "c", "d"],
    106:["normBkg", "B", "C", "b", "c"]
}
    return myBkgFunctions, myBkgSignalFunctions, myBkgParams
