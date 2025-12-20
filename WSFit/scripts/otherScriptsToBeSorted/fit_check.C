#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "RooFit.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooCustomizer.h"
#include "RooAddPdf.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "RooDataHist.h"
#include "RooArgList.h"
#include <iostream>
#include <map>
using namespace RooFit;

void fit_check(int category = 2, int verbose = 0, int shareAll = 0) {

    ROOT::Math::MinimizerOptions::SetDefaultTolerance(0.001);

    gROOT->SetBatch(kTRUE);

    // -----------------------------
    //  Paths and workspaces
    // -----------------------------
    TString ws_path0  = TString::Format("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws%d.root", category);
    TString ws_path10 = TString::Format("/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws1%d.root", category);

    TFile* f0  = TFile::Open(ws_path0);
    TFile* f10 = TFile::Open(ws_path10);
    RooWorkspace* ws0  = (RooWorkspace*)f0->Get("ws3");
    RooWorkspace* ws10 = (RooWorkspace*)f10->Get("ws3");

    // -----------------------------
    //  Variables
    // -----------------------------
    RooRealVar* x0  = ws0->var(TString::Format("dijet_mass_c%d", category));
    RooRealVar* x10 = ws10->var(TString::Format("dijet_mass_c1%d", category));
    double fit_min = 100, fit_max = 150;

    if (x0->getMin() != x10->getMin() || x0->getMax() != x10->getMax()) {
        std::cerr << "Min/Max values for the two categories are different!" << std::endl;
        return;
    }

    x0->setRange("fullRange_0", x0->getMin(), x0->getMax());
    x0->setRange("fitRangeLow_0", x0->getMin(), fit_min);
    x0->setRange("fitRangeHigh_0", fit_max, x0->getMax());
    x10->setRange("fullRange", x10->getMin(), x10->getMax());
    x10->setRange("fitRangeLow", x10->getMin(), fit_min);
    x10->setRange("fitRangeHigh", fit_max, x10->getMax());
    x10->setRange("fitRangeLow_0", x10->getMin(), fit_min);
    x10->setRange("fitRangeHigh_0", fit_min + 1e-5, x10->getMax());

    // -----------------------------
    //  PDFs
    // -----------------------------
    RooAbsPdf* pdf0 = ws0->pdf(TString::Format("env_pdf_Exponential_3_cat%d_exp3", category));

    RooRealVar* p1_0 = ws0->var(TString::Format("env_pdf_Exponential_3_cat%d_exp3_p1", category));

    std::cout << p1_0->GetName() << " : " << p1_0->getVal() << " +- " << p1_0->getError() << std::endl;
    std::cout << "Relative Error: " << p1_0->getError()/p1_0->getVal()*100 << " %" << std::endl;

    RooCustomizer customizer(*pdf0, TString::Format("pdf1%d_clone", category));
    customizer.replaceArg(*x0, *x10);

    RooRealVar* exp_p1_10 = nullptr;
    if (!shareAll) {
        exp_p1_10 = new RooRealVar("exp_p1_10", "slope of new exponential", -0.1, -10., 0.);
        customizer.replaceArg(*p1_0, *exp_p1_10);
    }

    RooAbsPdf* pdf10 = (RooAbsPdf*) customizer.build();

    RooAbsPdf* Z_model_0  = ws0->pdf(TString::Format("model_Z_c%d", category));
    RooAbsPdf* Z_model_10 = ws10->pdf(TString::Format("model_Z_c1%d", category));

    RooRealVar* model_Z_c0_norm  = ws0->var(TString::Format("model_Z_c%d_norm", category));
    RooRealVar* model_Z_c10_norm = ws10->var(TString::Format("model_Z_c1%d_norm", category));

    // -----------------------------
    //  Data histograms
    // -----------------------------
    RooAbsData* data0  = ws0->data(TString::Format("rooHist_data_cat%d", category));
    RooAbsData* data10 = ws10->data(TString::Format("rooHist_data_cat1%d", category));

    RooArgList pdf_list0(*pdf0, *Z_model_0);
    RooArgList coef_list0(data0->sumEntries(), *model_Z_c0_norm);
    RooAddPdf fullPdf_0(TString::Format("fullPdf_%d", category),
                         TString::Format("pdf%d + Z_model_%d", category, category),
                         pdf_list0, coef_list0);

    RooArgList pdf_list10(*pdf10, *Z_model_10);
    RooArgList coef_list10(data10->sumEntries(), *model_Z_c10_norm);
    RooAddPdf fullPdf_10(TString::Format("fullPdf_1%d", category),
                          TString::Format("pdf1%d + Z_model_1%d", category, category),
                          pdf_list10, coef_list10);

    // -----------------------------
    //  Category
    // -----------------------------
    RooCategory cat("channel", "channel");
    cat.defineType(TString::Format("cat%d", category));
    cat.defineType(TString::Format("cat1%d", category));

    // -----------------------------
    //  Simultaneous PDF
    // -----------------------------
    RooSimultaneous simPdf("simPdf", "simultaneous fit", cat);
    simPdf.addPdf(fullPdf_0, TString::Format("cat%d", category));
    simPdf.addPdf(fullPdf_10, TString::Format("cat1%d", category));

    // -----------------------------
    //  Fit
    // -----------------------------
    std::cout << "Previous fit" << std::endl;
    fullPdf_0.Print();
    std::cout << "New fit" << std::endl;

    RooFitResult* fitres0 = fullPdf_0.fitTo(*data0,
                                            Save(kTRUE),
                                            Minimizer("Minuit2", "minimize"),
                                            Strategy(2),
                                            Optimize(1),
                                            SumW2Error(kFALSE),
                                            Range("fitRangeLow_0,fitRangeHigh_0"));


    fitres0->Print();
}
