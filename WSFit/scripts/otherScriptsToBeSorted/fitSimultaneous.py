import ROOT
from ROOT import RooFit
import sys
from fitSimultaneousHelpers import draw_parameters_simple
ROOT.gROOT.SetBatch(True)
import argparse
# -----------------------------
#  Parser
# -----------------------------

parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
parser.add_argument("-c", "--category", type=int, help="Index of the workspace", default=2)
parser.add_argument("-v", "--verbose", type=int, help="Verbose (0 or 1)", default=0)
parser.add_argument("-shareAll", "--shareAll", type=int, help="Share all the parameters", default=0)
#parser.add_argument("-r", "--rebin", type=int, help="Rebinning", default=1)
args = parser.parse_args()


# -----------------------------
#  Workspaces
# -----------------------------

ws_path0  = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws%d.root"%args.category
ws_path10 = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws1%d.root"%args.category

f0  = ROOT.TFile.Open(ws_path0)
f10 = ROOT.TFile.Open(ws_path10)
ws0  = f0.Get("ws3")
ws10 = f10.Get("ws3")

# -----------------------------
#  Variables
# -----------------------------
x0  = ws0.var("dijet_mass_c%d"%args.category)
x10 = ws10.var("dijet_mass_c1%d"%args.category)
fit_min, fit_max = 100, 150

assert x0.getMin()==x10.getMin(), f"Min/Max values for the two categories are different: x{args.category} : [{x0.getMin()},{x0.getMax()}] x1{args.category} : [{x10.getMin()},{x10.getMax()}]"
assert x0.getMax()==x10.getMax(), f"Min/Max values for the two categories are different: x{args.category} : [{x0.getMin()},{x0.getMax()}] x1{args.category} : [{x10.getMin()},{x10.getMax()}]"
x0.setRange("fullRange_0", x0.getMin(), x0.getMax())
x0.setRange("fitRangeLow_0", x0.getMin(), fit_min)
x0.setRange("fitRangeHigh_0", fit_max, x0.getMax())
x10.setRange("fullRange", x10.getMin(), x10.getMax())
x10.setRange("fitRangeLow", x10.getMin(), fit_min)
x10.setRange("fitRangeHigh", fit_max, x10.getMax())
x10.setRange("fitRangeLow_0", x10.getMin(), fit_min)
x10.setRange("fitRangeHigh_0", fit_min+1e-5, x10.getMax())



Z_model_0_norm = ws0.var(f"model_Z_c{args.category}_norm")
Z_model_10_norm = ws10.var(f"model_Z_c1{args.category}_norm")
model_Z_c0_norm = ws0.var(f"model_Z_c{args.category}_norm")
model_Z_c10_norm = ws10.var(f"model_Z_c1{args.category}_norm")
H_model_0_norm = ws0.var(f"model_H_c{args.category}_norm")
H_model_10_norm = ws10.var(f"model_H_c1{args.category}_norm")


# -----------------------------
#  PDFs
# -----------------------------
Z_model_0 = ws0.pdf(f"model_Z_c{args.category}")
Z_model_10 = ws10.pdf(f"model_Z_c1{args.category}")
H_model_0 = ws0.pdf(f"model_H_c{args.category}")
H_model_10 = ws10.pdf(f"model_H_c1{args.category}")
pdf0  = ws0.pdf("env_pdf_Exponential_3_cat%d_exp3"%args.category)
parameters_0 = [
    ws0.var("env_pdf_Exponential_3_cat%d_exp3_p1" % args.category),
    ws0.var("env_pdf_Exponential_3_cat%d_exp3_p2" % args.category),
    ws0.var("env_pdf_Exponential_3_cat%d_exp3_f1" % args.category),
    ws0.var("env_pdf_Exponential_3_cat%d_z_norm" % args.category)
    ]
parameters_10 = [
    ws10.var("env_pdf_Exponential_3_cat%d_exp3_p1" % args.category),
    ws10.var("env_pdf_Exponential_3_cat%d_exp3_p2" % args.category),
    ws10.var("env_pdf_Exponential_3_cat%d_exp3_f1" % args.category),
    ws10.var("env_pdf_Exponential_3_cat%d_z_norm" % args.category)
    ]

for par in parameters_0:
    print(par.GetName(), " : ", par.getVal(), " +- ", par.getError())
print("Relative Error: ")
for par in parameters_0:
    print(par.GetName(), " : %.2f"%(par.getError()/par.getVal()*100))




# Clone pdf used for cat0 to fit cat10
customizer = ROOT.RooCustomizer(pdf0, "pdf1%d_clone"%args.category)
customizer.replaceArg(x0, x10)
# Build a proper RooAbsPdf
if not (args.shareAll):
    # If you dont want to share All Parameters some need to be replaced here
    exp_p2_10 = ROOT.RooRealVar("exp_p2_10", "slope of new exponential", -0.1, -10., 0.)
    exp_p1_10 = ROOT.RooRealVar("exp_p1_10", "slope of new exponential", -0.1, -10., 0.)
    exp_f1_10 = ROOT.RooRealVar("exp_f1_10", "fraction", 0.9, 0., 1)
    customizer.replaceArg(parameters_0[0], exp_p1_10)
    #customizer.replaceArg(parameters_0[1], exp_p2_10)
    #customizer.replaceArg(parameters_0[2], exp_f1_10)

pdf10 = customizer.build()



# -----------------------------
#  Data histograms
# -----------------------------
data0  = ws0.data(f"rooHist_data_cat{args.category}")
data10 = ws10.data(f"rooHist_data_cat1{args.category}")

pdf_list = ROOT.RooArgList(pdf0, Z_model_0)
#coef_list = ROOT.RooArgList(data0.sumEntries()-model_Z_c0_norm.getVal(), model_Z_c0_norm.getVal())  # Only the second term is absolute
coef_list = ROOT.RooArgList(1-parameters_0[3].getVal())  # Only the second term is absolute
fullPdf_0 = ROOT.RooAddPdf(f"fullPdf_{args.category}", f"pdf{args.category} + Z_model_{args.category}", pdf_list, coef_list)

pdf_list = ROOT.RooArgList(pdf10, Z_model_10)
coef_list = ROOT.RooArgList(1-parameters_10[3].getVal())  # Only the second term is absolute
fullPdf_10 = ROOT.RooAddPdf(f"fullPdf_1{args.category}", f"pdf1{args.category} + Z_model_1{args.category}", pdf_list, coef_list)





# -----------------------------
#  Category
# -----------------------------
cat = ROOT.RooCategory("category", "category")
cat.defineType(f"cat{args.category}")
cat.defineType(f"cat1{args.category}")

# -----------------------------
#  Simultaneous PDF
# -----------------------------
simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous fit", cat)
simPdf.addPdf(fullPdf_0, f"cat{args.category}")
simPdf.addPdf(fullPdf_10, f"cat1{args.category}")

# -----------------------------
#  Fit
# -----------------------------
# Note: RooSimultaneous automatically picks the correct dataset for each category
# Pass a list of datasets mapped by category
data_dict = ROOT.std.map('string,RooAbsData*')()
data_dict[f"cat{args.category}"]  = data0
data_dict[f"cat1{args.category}"] = data10

# Fit each dataset separately
# (RooSimultaneous uses Index to choose dataset, so we can build a combined dataset manually)
# Attempt to fit individually
#fitres0  = fullPdf_0.fitTo(data0, RooFit.Save(True),
#                      RooFit.Minimizer("Minuit2","minimize"),
#                      RooFit.Strategy(2),
#                      RooFit.Optimize(1),
#                      RooFit.PrintLevel(-1),
#                      #RooFit.Hesse(True),
#                      #RooFit.Minos(True),
#                      RooFit.SumW2Error(False),
#                      RooFit.Range("fitRangeLow_0,fitRangeHigh_0"),
#                      )
#
#fitres0.Print()

#for p in fitres0.floatParsFinal():
#    print(p.GetName(), p.getVal(), p.getError())


#fitres10 = pdf10.fitTo(data10, RooFit.Save(True))
print("Creating Data combination...")
combData = ROOT.RooDataSet("data_obs", "combined data",
                            ROOT.RooArgList(x0, x10),  # All observables
                            ROOT.RooFit.Index(cat),
                            ROOT.RooFit.Import(f"cat{args.category}", data0),
                            ROOT.RooFit.Import(f"cat1{args.category}", data10))
print("Simultaneous Fit...")
fitres = simPdf.fitTo(combData, 
                        RooFit.Save(True),
                        RooFit.Minimizer("Minuit2","minimize"),
                        RooFit.Strategy(2),
                        RooFit.Optimize(1),
                        RooFit.PrintLevel(-1),
                        RooFit.SumW2Error(False),
                        RooFit.Range("fitRangeLow_0,fitRangeHigh_0"),
                        RooFit.PrintLevel(-1))
print("\n"*10, "Fit Results")
fitres.Print()
print("\n"*10)
pdf0_postsimfit  = simPdf.getPdf(f"cat{args.category}")
pdf10_postsimfit  = simPdf.getPdf(f"cat1{args.category}")




# -----------------------------
#  Plot category 0
# -----------------------------
c0 = ROOT.TCanvas("c0", f"Category {args.category}", 600, 500)
pad1 = ROOT.TPad("pad1", "Main plot", 0, 0.3, 1, 1.0)
pad2 = ROOT.TPad("pad2", "Residuals", 0, 0.0, 1, 0.3)

pad1.SetBottomMargin(0.02)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.3)
pad1.Draw()
pad2.Draw()
pad1.cd()
frame0 = x0.frame(RooFit.Title(f"Category Pass"), RooFit.Range("fullRange"))
# Convert RooDataHist to TH1
hdata = data0.createHistogram("hdata", x0)
data0_rb = ROOT.RooDataHist("data0_rebinned", "rebinned data", ROOT.RooArgList(x0), hdata)

# Rebin by factor of 5
#hdata.Rebin(1)

data0_rb.plotOn(frame0, RooFit.MarkerSize(0.2), RooFit.Name("data_frame0"), RooFit.CutRange("fitRangeLow_0,fitRangeHigh_0"))
nData_low  = data0_rb.sumEntries("1", "fitRangeLow_0")
nData_high = data0_rb.sumEntries("1", "fitRangeHigh_0")
nData = nData_low + nData_high
pdf0_postsimfit.plotOn(frame0, RooFit.Name("pdf_frame0"),  RooFit.Normalization(nData, ROOT.RooAbsReal.NumEvent), RooFit.Range("fitRangeLow_0,fitRangeHigh_0"))
chi2_0 = frame0.chiSquare("pdf_frame0", "data_frame0")
#pdf0_postsimfit.plotOn(frame0, RooFit.VisualizeError(fitres, 2, False),
#                       RooFit.Normalization(1.0, ROOT.RooAbsReal.RelativeExpected)
#    #RooFit.FillColor(ROOT.kOrange),
#    #RooFit.FillStyle(3001),
#    #RooFit.Name("error_band")
#)
frame0.Draw()

txt0 = ROOT.TLatex(0.6, 0.85, f"#chi^{{2}}/ndf = {chi2_0:.4f}")
txt0.SetNDC()
txt0.SetTextSize(0.04)
txt0.Draw()
draw_parameters_simple(pdf0_postsimfit.getParameters(ROOT.RooArgSet(x0)), x=0.9, y=0.75)   
print(f"Reduced chi2(cat{args.category}) = {chi2_0:.5f}")
pad2.cd()

frame0.Print("P")
print("*&*\n"*10)


# Compute residuals (difference) or pulls (normalized difference)
residHist_0 = frame0.pullHist("data_frame0", "pdf_frame0")  # or use frame0.pullHist() for pulls
# Build a frame for residuals
frame_resid = x0.frame(RooFit.Title("Residuals"))
residHist_0.SetMarkerSize(0.2)
#residHist.Print("V")
print("\n*"*20)
frame_resid.addPlotable(residHist_0, "P")  # P = markers
frame_resid.GetYaxis().SetTitle("Data - Fit")
frame_resid.GetYaxis().SetTitleSize(0.08)
frame_resid.GetYaxis().SetLabelSize(0.08)
frame_resid.GetXaxis().SetTitleSize(0.1)
frame_resid.GetXaxis().SetLabelSize(0.1)
frame_resid.Draw()
line0 = ROOT.TLine(x0.getMin(), 0, x0.getMax(), 0)
line0.SetLineStyle(2)
line0.Draw("same")

c0.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat{args.category}/plots/fit_cat{args.category}.png")
c0.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat1{args.category}/plots/fit_cat{args.category}.png")

if args.verbose:
    for i in range(data0_rb.numEntries()):
        # Get the i-th entry as a RooArgSet
        vals = data0_rb.get(i)

        # Get the value of your variable (here: x0)
        x_val = vals.getRealValue("dijet_mass_c%d" % args.category)

        # Get the weight of this bin
        weight = data0_rb.weight(i)

        print(f"Bin {i}: x = {x_val:.2f}, weight = {weight:.2f}")

    residHist_0.Print("V") 






















# -----------------------------
#  Plot category 10
# -----------------------------
c10 = ROOT.TCanvas(f"c1{args.category}", f"Category 1{args.category}", 600, 500)
pad10_1 = ROOT.TPad("pad1", "Main plot", 0, 0.3, 1, 1.0)
pad10_2 = ROOT.TPad("pad2", "Residuals", 0, 0.0, 1, 0.3)
pad10_1.SetBottomMargin(0.02)
pad10_2.SetTopMargin(0.05)
pad10_2.SetBottomMargin(0.3)
pad10_1.Draw()
pad10_2.Draw()
pad10_1.cd()

frame10 = x10.frame(RooFit.Title(f"Category Fail"), RooFit.Range("fullRange"))

hdata = data10.createHistogram("hdata", x10)
data10_rb = ROOT.RooDataHist("data0_rebinned", "rebinned data", ROOT.RooArgList(x10), hdata)


data10_rb.plotOn(frame10, RooFit.MarkerSize(0.2),)
pdf10_postsimfit.plotOn(frame10, RooFit.Range("fullRange"))
frame10.Draw()

chi2_10 = frame10.chiSquare()
txt10 = ROOT.TLatex(0.6, 0.85, f"#chi^{{2}}/ndf = {chi2_10:.3f}")
txt10.SetNDC()
txt10.SetTextSize(0.04)
txt10.Draw()


print(f"Reduced chi2(cat1{args.category}) = {chi2_10:.5f}")
pad10_2.cd()


resid10 = frame10.pullHist()
resid10.SetMarkerStyle(20)
resid10.SetMarkerSize(0.4)
frame_r10 = x10.frame(RooFit.Title(""))
frame_r10.addPlotable(resid10, "P")
frame_r10.GetYaxis().SetTitle("Data - Fit")
frame_r10.GetYaxis().SetTitleSize(0.08)
frame_r10.GetYaxis().SetLabelSize(0.08)
frame_r10.GetXaxis().SetTitleSize(0.1)
frame_r10.GetXaxis().SetLabelSize(0.1)
frame_r10.Draw()
line10 = ROOT.TLine(x10.getMin(), 0, x10.getMax(), 0)
line10.SetLineStyle(2)
line10.Draw("same")


c10.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat{args.category}/plots/fit_cat1{args.category}.png")
c10.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat1{args.category}/plots/fit_cat1{args.category}.png")






















# -----------------------------
#  Combined canvas (optional)
# -----------------------------
# Create the canvas
cAll = ROOT.TCanvas("cAll", "Fits with residuals", 1200, 700)

# --- Define pad sizes
main_height = 0.7  # fraction for main plots
res_height  = 0.3  # fraction for residuals

# --- Left category (cat0)
pad0_main  = ROOT.TPad("pad0_main",  "Main 0",  0.0, res_height, 0.5, 1.0)
pad0_resid = ROOT.TPad("pad0_resid", "Resid 0", 0.0, 0.0,        0.5, res_height)

# --- Right category (cat10)
pad1_main  = ROOT.TPad("pad1_main",  "Main 1",  0.5, res_height, 1.0, 1.0)
pad1_resid = ROOT.TPad("pad1_resid", "Resid 1", 0.5, 0.0,        1.0, res_height)

# --- Margins
for pad in [pad0_main, pad1_main]:
    pad.SetBottomMargin(0.02)
for pad in [pad0_resid, pad1_resid]:
    pad.SetTopMargin(0.05)
    pad.SetBottomMargin(0.3)

# Draw pads
pad0_main.Draw()
pad0_resid.Draw()
pad1_main.Draw()
pad1_resid.Draw()

# -----------------------------

pad0_main.cd()
frame0.Draw()
draw_parameters_simple(pdf0_postsimfit.getParameters(ROOT.RooArgSet(x0)), x=0.9, y=0.75)
chi2_0 = frame0.chiSquare()
txt0 = ROOT.TLatex(0.6, 0.85, f"#chi^{{2}}/ndf = {chi2_0:.3f}")
txt0.SetNDC()
txt0.SetTextSize(0.04)
txt0.Draw()

pad0_resid.cd()
frame_r0 = x0.frame(RooFit.Title("Residuals"))

resid0 = frame0.pullHist("data_frame0", "pdf_frame0")
resid0.SetMarkerStyle(20)
resid0.SetMarkerSize(0.4)
frame_r0.addPlotable(resid0, "P")
frame_r0.GetYaxis().SetTitle("Data - Fit")
frame_r0.GetYaxis().SetTitleSize(0.08)
frame_r0.GetYaxis().SetLabelSize(0.08)
frame_r0.GetXaxis().SetTitleSize(0.1)
frame_r0.GetXaxis().SetLabelSize(0.1)
frame_r0.Draw()

# Optional line at y=0
line0 = ROOT.TLine(x0.getMin(), 0, x0.getMax(), 0)
line0.SetLineStyle(2)
line0.Draw("same")

# ==========================================================
# Plot right category (cat10)
# ==========================================================
pad1_main.cd()
frame10.Draw()
draw_parameters_simple(pdf10_postsimfit.getParameters(ROOT.RooArgSet(x10)), x=0.9, y=0.75)

chi2_10 = frame10.chiSquare()
txt10 = ROOT.TLatex(0.6, 0.85, f"#chi^{{2}}/ndf = {chi2_10:.3f}")
txt10.SetNDC()
txt10.SetTextSize(0.04)
txt10.Draw()

pad1_resid.cd()
resid10 = frame10.pullHist()
resid10.SetMarkerStyle(20)
resid10.SetMarkerSize(0.4)
frame_r10 = x10.frame(RooFit.Title(""))
frame_r10.addPlotable(resid10, "P")
frame_r10.GetYaxis().SetTitle("Data - Fit")
frame_r10.GetYaxis().SetTitleSize(0.08)
frame_r10.GetYaxis().SetLabelSize(0.08)
frame_r10.GetXaxis().SetTitleSize(0.1)
frame_r10.GetXaxis().SetLabelSize(0.1)
frame_r10.Draw()
line10 = ROOT.TLine(x10.getMin(), 0, x10.getMax(), 0)
line10.SetLineStyle(2)
line10.Draw("same")
cAll.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat{args.category}/plots/fits_cat{args.category}_cat1{args.category}.png")
cAll.SaveAs(f"/t3home/gcelotto/ggHbb/WSFit/output/cat1{args.category}/plots/fits_cat{args.category}_cat1{args.category}.png")



for par in parameters_0:
    print(par.GetName(), " : %.2f"%(par.getError()/par.getVal()*100))




print("Preparing the worksapce")

nBkg = ws0.var("CMS_hgg_0_2016_13TeV_bkgshape_noZ_norm")
bkg_norm_0 = ROOT.RooRealVar("pdf_cat%d_norm"%args.category, "pdf_cat%d_norm"%args.category, nBkg.getVal(), nBkg.getVal()*0.5, nBkg.getVal()*1.5)
nBkg_10 = ws10.var("CMS_hgg_0_2016_13TeV_bkgshape_noZ_norm")
bkg_norm_10 = ROOT.RooRealVar("pdf_cat1%d_norm"%args.category, "pdf_cat1%d_norm"%args.category, nBkg_10.getVal(), nBkg_10.getVal()*0.5, nBkg_10.getVal()*1.5)
print(bkg_norm_10.GetName())


simPdf_bkg_only = fullPdf_0.pdfList().at(0)
simPdf_bkg_only.SetName("pdf_cat%d"%args.category)
ws_out = ROOT.RooWorkspace("w", "Simultaneous fit workspace")
getattr(ws_out, "import")(simPdf_bkg_only)
getattr(ws_out, "import")(data0)
getattr(ws_out, "import")(Z_model_0)
getattr(ws_out, "import")(bkg_norm_0)
getattr(ws_out, "import")(H_model_0)
getattr(ws_out, "import")(H_model_0_norm)
getattr(ws_out, "import")(Z_model_0_norm)
output_path = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnrichedShared/ws%d.root"%args.category
ws_out.writeToFile(output_path)


nBkg = ws10.var("CMS_hgg_0_2016_13TeV_bkgshape_noZ_norm")
bkg_norm = ROOT.RooRealVar("pdf_cat%d_norm"%args.category, "pdf_cat%d_norm"%args.category, nBkg.getVal(), nBkg.getVal()*0.5, nBkg.getVal()*1.5)
simPdf_bkg_only = fullPdf_10.pdfList().at(0)
simPdf_bkg_only.SetName("pdf_cat1%d"%args.category)
ws_out = ROOT.RooWorkspace("w", "Simultaneous fit workspace")
getattr(ws_out, "import")(simPdf_bkg_only)
getattr(ws_out, "import")(data10)
getattr(ws_out, "import")(bkg_norm_10)
getattr(ws_out, "import")(Z_model_10)
getattr(ws_out, "import")(H_model_10)
getattr(ws_out, "import")(H_model_10_norm)
getattr(ws_out, "import")(Z_model_10_norm)
output_path = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnrichedShared/ws1%d.root"%args.category
ws_out.writeToFile(output_path)
