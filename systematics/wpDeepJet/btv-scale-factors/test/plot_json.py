import correctionlib
from correctionlib.highlevel import model_auto, open_auto
import yaml
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import mplhep.cms
plt.style.use(mplhep.style.CMS)
plt.rcParams.update({"font.size": 38})

# load database
test_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(test_dir, "plot_db.yml"), "r") as yf:
    db = yaml.load(yf, Loader=yaml.BaseLoader)
# load index file
with open(os.path.join(test_dir, "index.php"), "r") as idxf:
    php = idxf.readlines()
with open(os.path.join(test_dir, "index.html"), "r") as htmlf:
    html = htmlf.readlines()

def init_dir(dir, do_php=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    if do_php:
        with open(os.path.join(dir, "index.php"), "w") as idxf:
            idxf.write("".join(php))

def delete_empty_dir(dir):
    files = glob.glob(f"{dir}/*")
    n_files = len(files)
    n_php_files = len(glob.glob(f"{dir}/*.php"))
    if n_files == n_php_files:
        for f in files:
            os.remove(f) 
        os.rmdir(dir)

def get_correction_values(cset, cname, dim):
    correction = None
    for c in cset.corrections:
        if c.name == cname:
            correction = c
            break

    if correction:
        input_dim = correction.summary()[1][dim]
        return input_dim.values
    return []

# directories
in_dir = sys.argv[1]
base_dir = os.path.dirname(os.path.abspath(in_dir))

# get campaign name
campaign = os.path.basename(base_dir)

# make plot dir
plot_base_dir = os.path.join(os.path.dirname(base_dir), "plots")
init_dir(plot_base_dir)

# make outdir
out_dir = os.path.join(plot_base_dir, campaign)
init_dir(out_dir)


# some common labels
label_dict = {
    "sys8":    {"label": "System8", "symbol": "d"},
    "ptrel":   {"label": "$p_{T}$ rel.", "symbol": "o"},
    "ltsv":    {"label": "LTSV", "symbol": "s"},
    "tnp":     {"label": "Tag&Probe", "symbol": "^"},
    "kinfit":  {"label": "kin. fit", "symbol": "v"},

    "comb":    {"label": "comb", "color": "C0", "style": "--"},
    "mujets":  {"label": "mujets", "color": "C2", "style": ":"},

    "wc":      {"label": "OS$-$SS W$+$c", "flav": 4},
    "light":   {"label": "Negative Tag", "flav": 0},
    }
wp_dict = {
    "XXT":     {"symbol": "d", "style": "--", "color": "C4"},
    "XT":      {"symbol": "^", "style": "--", "color": "C2"},
    "T":       {"symbol": "s", "style": "-.", "color": "C1"},
    "M":       {"symbol": "v", "style": "-.", "color": "C0"},
    "L":       {"symbol": "o", "style": ":",  "color": "C3"},
}
tagger_dict = {
    "robustParticleTransformer": "robustParT"
    }

# lumi label info
# from https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis
campaign_lumi = {
    "2022_Summer22": 4.953+2.922, 
    "2022_Summer22EE": 5.672+17.610+3.055,
    "2023_Summer23": 17.981,
    "2023_Summer23BPix": 9.516
    }
lumi = campaign_lumi.get(campaign,0.)
lumi_label = f"{campaign} (${lumi:.1f}\,fb^{{-1}}$)"

def get_tagger_label(tagger):
    return tagger_dict.get(tagger, tagger)

def get_labels(method):
    color = label_dict[method].get("color", None)
    symbol = label_dict[method].get("symbol", None)
    label = label_dict[method].get("label", method)
    flav = label_dict[method].get("flav", 5)
    return color, symbol, label, int(flav)

def get_labels_wp(wp):
    color = wp_dict[wp].get("color", None)
    symbol = wp_dict[wp].get("symbol", None)
    style = wp_dict[wp].get("style", None)
    return color, symbol, style

def flav_to_text(flav):
    if flav==5: return "b"
    elif flav==4: return "c"
    elif flav==0: return "light"
    else: 
        print(f"invalid flavour {flav}")
        exit

## function for working point tables
def working_points(**kwargs):
    '''
    working_points_ctagging:
        function: "working_points"
        title: "c tagging working points"
        input: "{tagger}_wp_values"
        wp: ["L", "M", "T", "XT", "XXT"]
        axes: ["CvB", "CvL"]
    '''
    # if no axes are given no argument has to be passed to the evaluator
    # will be used to access the elements and as table titles
    axes = kwargs.get("axes", [])
    no_arg = len(axes) == 0
    if no_arg:
        axes = [""]
    
    wps = kwargs.get("wp", [])
    title = kwargs.get("title", "")
    
    taggers = kwargs["taggers"]
    cset = kwargs["cset"]
    plot_dir = kwargs["plot_dir"]

    tables = []
    tables.append(f"## {title}")
    for tagger in taggers:
        # table header
        tables.append(f"### {tagger} ")
        tables.append(f"| WP | {' | '.join(wps)} | ")
        tables.append(f"| -- | {' | '.join(['--']*len(wps))} | ")

        corr = cset[f"{tagger}_wp_values"]
        for ax in axes:
            if no_arg:
                wp_val = [ str(corr.evaluate(wp)) for wp in wps ]
            else:
                wp_val = [ str(corr.evaluate(wp, ax)) for wp in wps ]
            tables.append(f"| {ax} | {' | '.join(wp_val)} | ")

        tables.append("-----")

    # add formatted md code to html file
    table_str = "\n".join(tables)
    out_html = "".join([l.replace("<!-- placeholder -->", table_str) for l in html])
    
    # save html
    out_file = os.path.join(plot_dir, "index.html")
    with open(out_file, "w") as f:
        f.write(out_html)
    
def itFit_c(**kwargs):
    '''
    shape_CvB:
        function: "itFit_c"
        input: "{tagger}_shape"
        xvariable: "cvl"
        yvariable: "cvb"
        yvalues: [0.1, 0.3, 0.5, 0.7, 0.9]
        systs: ["TotalUnc"]
    '''
    yvals = kwargs.get("yvalues", [])
    if kwargs.get("xvariable", None) == "cvl":
        scan_cvb = False
    elif kwargs.get("xvariable", None) == "cvb":
        scan_cvb = True
    else:
        # invalid configuration
        return

    flavs = kwargs.get("flavs", [])
    systs = kwargs.get("systs", [])
    
    taggers = kwargs["taggers"]
    cset = kwargs["cset"]

    plot_dir = kwargs["plot_dir"]

    x = np.arange(0., 1.01, 0.01)
    for tagger in taggers:
        corr = cset[f"{tagger}_shape"]
        for flav in flavs:
            for yval in yvals:
                # one plot

                fig, ax = plt.subplots(1, 1, figsize=(16, 14))
                ax.plot([x[0], x[-1]], [1, 1], c="black", lw=1)

                hdls = []
                ls = []
        
                flav_text = flav_to_text(int(flav))

                # nominal sf
                if scan_cvb:
                    nom = corr.evaluate("central", int(flav), yval, x)        
                else:
                    nom = corr.evaluate("central", int(flav), x, yval)

                hdl, = ax.plot(x, nom, lw=3, marker="", ls="-", color="black")
                hdls.append(hdl)
                ls.append(f"nominal {flav_text} jet SF")

                for v in systs:
                    if scan_cvb:
                        up = corr.evaluate(f"up_{v}", int(flav), yval, x)
                        dn = corr.evaluate(f"down_{v}", int(flav), yval, x)
                    else:
                        up = corr.evaluate(f"up_{v}", int(flav), x, yval)
                        dn = corr.evaluate(f"down_{v}", int(flav), x, yval)
                    
                    hdl, = ax.plot(x, up, lw=3, marker="", ls="-.")
                    hdls.append(hdl)
                    ls.append(f"variation: {v} (up)")

                    hdl, = ax.plot(x, dn, lw=3, marker="", ls=":")
                    hdls.append(hdl)
                    ls.append(f"variation: {v} (down")

                # labels and legends
                tagger_label = get_tagger_label(tagger)
                if scan_cvb:
                    title = f"{tagger_label} SFs ($CvL={float(yval):.1f}$)"
                    ax.set_xlabel("CvB discriminant")                   
                    out_title = f"cvl{float(yval):.1f}"
                else:
                    title = f"{tagger_label} SFs ($CvB={float(yval):.1f}$)"
                    ax.set_xlabel("CvL discriminant")
                    out_title = f"cvb{float(yval):.1f}"

                l2 = ax.legend(hdls, ls, loc="upper left", ncol=2,
                    title=title, title_fontsize="small")
                plt.gca().add_artist(l2)

                ax.set_ylabel(f"{flav_text} jet SF")
                mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)

                # make pretty
                ax.set_xlim(x[0], x[-1])
                x_dist = ax.get_ylim()[1]-ax.get_ylim()[0]
                ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+x_dist*1.25)
                fig.tight_layout()
                plt.grid(True, axis="x")

                # save
                out_file = os.path.join(plot_dir, f"{tagger}_c_shape_{flav_text}_{out_title}.pdf")
                fig.savefig(out_file)
                fig.savefig(out_file.replace(".pdf",".png"))
                plt.close()
            

def itFit_disc(**kwargs):
    '''
    shape_disc_light:
        function: "itFit_disc"
        pt_values: [30, 100]
        eta_values: [0]
        flav: 0
        systs: ["hf", "lf"]
    '''
    pts = kwargs.get("pt_values", [])
    etas = kwargs.get("eta_values", [])
    flavs = kwargs.get("flavs", [])
    systs = kwargs.get("systs", [])

    taggers = kwargs["taggers"]
    cset = kwargs["cset"]

    plot_dir = kwargs["plot_dir"]
    
    x = np.arange(0., 1.01, 0.01)      
    for tagger in taggers:
        corr = cset[f"{tagger}_shape"]
        for pt in pts:
            for eta in etas:
                for flav in flavs:
                    # one plot
                    print(f"Plotting shape correction for {tagger} with flav={flav}, pt={pt}, eta={eta}")
                    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
                    ax.plot([x[0], x[-1]], [1, 1], c="black", lw=1)

                    hdls = []
                    ls = []

                    flav_text = flav_to_text(int(flav))
                    
                    # nominal sf
                    nom = corr.evaluate("central", int(flav), float(eta), float(pt), x)
                    hdl, = ax.plot(x, nom, lw=3, marker="", ls="-", color="black")
                    hdls.append(hdl)
                    ls.append(f"nominal {flav_text} jet SF")

                    for v in systs:
                        up = corr.evaluate(f"up_{v}", int(flav), float(eta), float(pt), x)
                        hdl, = ax.plot(x, up, lw=3, marker="", ls="-.")
                        hdls.append(hdl)
                        ls.append(f"variation: {v} (up)")

                        dn = corr.evaluate(f"down_{v}", int(flav), float(eta), float(pt), x)
                        hdl, = ax.plot(x, dn, lw=3, marker="", ls=":", color=hdl.get_color())
                        hdls.append(hdl)
                        ls.append(f"variation: {v} (down)")
                        
                    # labels and legends
                    tagger_label = get_tagger_label(tagger)
                    l2 = ax.legend(hdls, ls, loc="upper left", ncol=2,
                        title=f"{tagger_label} SFs ($p_{{T}}={float(pt):.0f}$ GeV, $|\eta|={float(eta):.0f}$)", title_fontsize="small")
                    plt.gca().add_artist(l2)

                    ax.set_ylabel(f"{flav_text} jet SF")
                    ax.set_xlabel("b jet discriminant")
                
                    mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)
                    
                    # make pretty
                    ax.set_xlim(x[0], x[-1])
                    x_dist = ax.get_ylim()[1]-ax.get_ylim()[0]
                    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+x_dist*1.4)
                    fig.tight_layout()
                    plt.grid(True, axis="x")
                
                    # save
                    out_file = os.path.join(plot_dir, f"{tagger}_shape_{flav_text}_pt{pt}_eta{eta}.pdf")
                    fig.savefig(out_file)
                    fig.savefig(out_file.replace(".pdf",".png"))
                    plt.close()
                

def itFit_pt(**kwargs):
    '''
    shape_pt_b:
        function: "itFit_pt"
        disc_values: [0.7, 0.9]
        eta_values: [0]
        flav: 5
        systs: ["hf", "lf"]
    '''
    discs = kwargs.get("disc_values", [])
    etas = kwargs.get("eta_values", [])
    flavs = kwargs.get("flavs", [])
    systs = kwargs.get("systs", [])

    taggers = kwargs["taggers"]
    cset = kwargs["cset"]

    plot_dir = kwargs["plot_dir"]
    
    pt = np.arange(20., 300., 0.1)      
    for tagger in taggers:
        corr = cset[f"{tagger}_shape"]
        for x in discs:
            for eta in etas:
                for flav in flavs:
                    # one plot
                    print(f"Plotting shape correction for {tagger} with flav={flav}, disc={x}, eta={eta}")
                    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
                    ax.plot([pt[0], pt[-1]], [1, 1], c="black", lw=1)

                    hdls = []
                    ls = []

                    flav_text = flav_to_text(int(flav))
                    
                    # nominal sf
                    nom = corr.evaluate("central", int(flav), float(eta), pt, float(x))
                    hdl, = ax.plot(pt, nom, lw=3, marker="", ls="-", color="black")
                    hdls.append(hdl)
                    ls.append(f"nominal {flav_text} jet SF")

                    for v in systs:
                        up = corr.evaluate(f"up_{v}", int(flav), float(eta), pt, float(x))
                        hdl, = ax.plot(pt, up, lw=3, marker="", ls="-.")
                        hdls.append(hdl)
                        ls.append(f"variation: {v} (up)")

                        dn = corr.evaluate(f"down_{v}", int(flav), float(eta), pt, float(x))
                        hdl, = ax.plot(pt, dn, lw=3, marker="", ls=":", color=hdl.get_color())
                        hdls.append(hdl)
                        ls.append(f"variation: {v} (down)")
                        
                    # labels and legends
                    tagger_label = get_tagger_label(tagger)
                    l2 = ax.legend(hdls, ls, loc="upper left", ncol=2,
                        title=f"{tagger_label} SFs ($discr. value={x}$)", title_fontsize="small")
                    plt.gca().add_artist(l2)

                    ax.set_ylabel(f"{flav_text} jet SF")
                    ax.set_xlabel("jet $p_{T}$ (GeV)")
                
                    mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)
                    
                    # make pretty
                    ax.set_xscale("log")
                    ax.set_xlim(pt[0], pt[-1])
                    x_dist = ax.get_ylim()[1]-ax.get_ylim()[0]
                    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+x_dist*1.4)
                    fig.tight_layout()
                    plt.grid(True, axis="x")
                    ticks = [20, 40, 60, 80, 120, 200, 300]
                    plt.xticks(ticks, ticks)
                    plt.minorticks_off()

                    # save
                    out_file = os.path.join(plot_dir, f"{tagger}_shape_{flav_text}_disc{x}_eta{eta}.pdf")
                    fig.savefig(out_file)
                    fig.savefig(out_file.replace(".pdf",".png"))
                    plt.close()
                


def fixedWP_comb(**kwargs):
    wps = kwargs.get("wp", [])
    taggers = kwargs.get("taggers", [])
    comb = kwargs.get("combination", "comb")
    cset = kwargs["cset"]
    plot_dir = kwargs["plot_dir"]

    pt_bins = np.array([int(x) for x in kwargs.get("pt_bins", [])])
    if len(pt_bins) < 2:
        print(f"not enough pt bins specified: {pt_bins}")
        return
    pt_centers = (pt_bins[1:]+pt_bins[:-1])/2
    pt_widths = (pt_bins[1:]-pt_bins[:-1])/2

    # load file with SFs from individual methods
    # file should be located at XXX_methods.json
    in_json = kwargs.get("in_json", None)
    methods = kwargs.get("methods", [])
    add_method_lines = False
    if len(methods) > 0 and in_json:
        methods_file = in_json.replace("btagging_v","btagging_methods_v")
        if not os.path.exists(methods_file):
            print(f"No file for SF methods found at {methods_file}")
            methods = []
            methods_list = []
        else:
            methods_cset = correctionlib.CorrectionSet.from_file(methods_file)
            methods_dict = model_auto(open_auto(methods_file))
            methods_list = list(methods_cset)
            add_method_lines = True
    
    def get_max_pt_bin(sf):
        # get the last bin of the SF
        max_pt_bin = 0
        while np.unique(sf[max_pt_bin:]).size != 1:
            max_pt_bin += 1
        return max_pt_bin+1

    for tagger in taggers:
        # have to be in db list and methods.json
        available_methods = [
            m for m in methods if f"{tagger}_{m}" in methods_list]
        methods_wps = {m: get_correction_values(methods_dict, f"{tagger}_{m}", "working_point")
                        for m in available_methods}

        for wp in wps:
            print(f"Plotting fixedWP SF combination '{comb}' for {tagger} WP '{wp}'.")
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            ax.plot([pt_bins[0], pt_bins[-1]], [1, 1], c="black", lw=2)

            m_legends = []
            m_labels = []
            # plot available methods
            for i, m in enumerate(available_methods):
                if not wp in methods_wps[m]:
                    print(f"WP {wp} not defined for {m}")
                    continue

                color, symbol, label, flav = get_labels(m)

                # read method SF values
                c = methods_cset[f"{tagger}_{m}"]
                sf = c.evaluate("central", wp, flav, 0., pt_centers)                   
                try:
                    sf_up = c.evaluate("up", wp, flav, 0., pt_centers)                   
                    sf_dn = c.evaluate("down", wp, flav, 0., pt_centers)
                except:
                    sf_up = sf_dn = sf
                n = get_max_pt_bin(sf)
            
    
                # move centers a bit
                offset = i*(-1)**i
                centers = pt_centers[:n]+offset
                
                all_sf = np.array([sf, sf_dn, sf_up])
                sf_max = np.amax(all_sf, axis=0)
                sf_min = np.amin(all_sf, axis=0)

                # plot it
                hdl = ax.errorbar(centers, sf[:n],
                    yerr=(sf[:n]-sf_min[:n], sf_max[:n]-sf[:n]),
                    xerr=(pt_widths[:n]+offset, pt_widths[:n]-offset),
                    ls="", ms=9, mew=2, lw=2,
                    marker=symbol, color=color)

                # add legend info
                m_legends.append(hdl)
                m_labels.append(label)
            
            # legend for methods
            if f"{tagger}_{comb}" in list(cset):
                l1 = ax.legend(m_legends, m_labels, loc="upper right",
                            ncol=1 if len(m_legends)<4 else 2)
            else:
                l1 = ax.legend(m_legends, m_labels, loc="upper right",
                            title=f"{tagger} {wp} WP", title_fontsize="small",
                            ncol=1 if len(m_legends)<4 else 2)
            plt.gca().add_artist(l1)

            # add SF combination if available
            if f"{tagger}_{comb}" in list(cset):
                _, _, _, flav = get_labels(comb)

                # read combined SF
                c = cset[f"{tagger}_{comb}"]
                pt_range = np.arange(pt_bins[0], pt_bins[-1], 1)
                sf = c.evaluate("central", wp, flav, 0., pt_range)
                sf_up = c.evaluate("up", wp, flav, 0., pt_range)
                sf_dn = c.evaluate("down", wp, flav, 0., pt_range)

                # plot combined SF
                f_hdl, = ax.plot(pt_range, sf, c="black", lw=3, marker="")
                s_hdl = ax.fill_between(pt_range, sf_dn, sf_up, alpha=0.3, color="black")

                # legend for sf
                tagger_label = get_tagger_label(tagger)
                l2 = ax.legend([f_hdl, s_hdl], ["fit", "fit $\pm$ (stat $\oplus$ syst)"], loc="upper left",
                    title=f"{tagger_label} {wp} WP ({comb})", title_fontsize="small")
                plt.gca().add_artist(l2)


            # set axis labels
            ax.set_ylabel("$SF_{b}$")
            ax.set_xlabel("jet $p_{T}$ (GeV)")
            mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)

            # make pretty
            ax.set_xscale("log")
            ax.set_xlim(pt_bins[0], pt_bins[-1])
            rng = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+rng*1.5)
            fig.tight_layout()
            plt.xticks(pt_bins, pt_bins)
            plt.grid(True, axis="x")

            # save
            out_file = os.path.join(plot_dir, f"{tagger}_{campaign}_{wp}WP.pdf")
            fig.savefig(out_file)
            fig.savefig(out_file.replace(".pdf",".png"))
            plt.close()



def fixedWP_SF(**kwargs):
    wps = kwargs.get("wp", [])
    taggers = kwargs.get("taggers", [])

    methods = kwargs.get("method", "comb")
    if type(methods) == str:
        methods = [methods]

    cset = kwargs["cset"]
    plot_dir = kwargs["plot_dir"]
    json_file = os.path.basename(kwargs["in_json"]).replace(".json","")
    json_dict = model_auto(open_auto(kwargs["in_json"]))


    pt_range = kwargs.get("pt_range", None)
    pt_bins = kwargs.get("pt_bins", None)

    binned = False
    if pt_bins: 
        binned = True
        pt_bins = np.array([int(x) for x in pt_bins])
        if len(pt_bins) < 2:
            print(f"not enough pt bins specified: {pt_bins}")
            return

        bin_c = (pt_bins[1:]+pt_bins[:-1])/2.
        bin_w = (pt_bins[1:]-pt_bins[:-1])/2.
        pt_range = np.array([pt_bins[0], pt_bins[-1]])
    elif pt_range:
        pt_range = np.array([int(x) for x in pt_range])
        pt_bins = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000, 1400]
        pt_bins = [x for x in pt_bins if x>=pt_range[0] and x<=pt_range[1]]
        bin_c = np.arange(pt_range[0], pt_range[1], 1.)
    else:
        print(f"Need to specify either pt_range or pt_bins")
        return


    for tagger in taggers:

        for method in methods:
            print(f"Plotting fixedWP SFs of '{method}' for {tagger}.")
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))
            ax.plot([pt_range[0], pt_range[-1]], [1, 1], c="black", lw=1)

            hdls = []
            ls = []

            _, _, label, flav = get_labels(method)
            flav_text = flav_to_text(int(flav))
        
            c = f"{tagger}_{method}"
            if json_file == "ctagging":
                c = f"{tagger}_wp"
            if not c in list(cset): continue
            corr = cset[c]

            available_wps = get_correction_values(
                json_dict, f"{tagger}_{method}", "working_point")

            for i, wp in enumerate(wps):
                if not wp in available_wps:
                    print(f"WP {wp} not available for {method}")
                    continue
                c, m, s = get_labels_wp(wp)
                if json_file == "ctagging":
                    sf = corr.evaluate("central", method, wp, flav, 0., bin_c)
                    try:
                        sf_up = corr.evaluate("up", method, wp, flav, 0., bin_c)
                        sf_dn = corr.evaluate("down", method, wp, flav, 0., bin_c)
                    except:
                        sf_up = sf_dn = sf
                else:
                    sf = corr.evaluate("central", wp, flav, 0., bin_c)
                    try:
                        sf_up = corr.evaluate("up", wp, flav, 0., bin_c)
                        sf_dn = corr.evaluate("down", wp, flav, 0., bin_c)
                    except:
                        sf_up = sf_dn = sf

                if binned:
                    all_sf = np.array([sf, sf_dn, sf_up])
                    sf_max = np.amax(all_sf, axis=0)
                    sf_min = np.amin(all_sf, axis=0)
                    # move the bin centers a bit
                    offset = i*(-1)**i
                    
                    hdl = ax.errorbar(bin_c+offset, sf,
                        yerr=(sf-sf_min, sf_max-sf),
                        xerr=(bin_w+offset, bin_w-offset),
                        ls="", ms=9, mew=2, lw=2,
                        marker=m, color=c)
                else:
                    hdl, = ax.plot(bin_c, sf, lw=3, marker="", ls=s, color=c)
                    sys_hdl = ax.fill_between(bin_c, sf_dn, sf_up, alpha=0.3, color=c)

                hdls.append(hdl)
                ls.append(f"{flav_text} jet SF ({wp} WP)")

            # labels and legends
            tagger_label = get_tagger_label(tagger)
            l2 = ax.legend(hdls, ls, loc="upper left", ncol=2,
                title=f"{tagger_label} ({label})", title_fontsize="small")
            plt.gca().add_artist(l2)

            ax.set_ylabel(f"$SF_{{{flav_text}}}$")
            ax.set_xlabel("jet $p_{T}$ (GeV)")

            mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)

            # make pretty
            ax.set_xscale("log")
            ax.set_xlim(pt_range[0], pt_range[-1])
            r = ax.get_ylim()[1]-ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+r*1.5)
            fig.tight_layout()
            plt.xticks(pt_bins, pt_bins)
            plt.minorticks_off()
            plt.grid(True, axis="x")

            # save
            out_file = os.path.join(plot_dir, f"{tagger}_{method}_wps.pdf")
            fig.savefig(out_file)
            fig.savefig(out_file.replace(".pdf",".png"))
            plt.close()

        
        # additional plots to compare different methods
        if len(methods) < 2: 
            continue
        
        available_wps = {method: get_correction_values(
            json_dict, f"{tagger}_{method}", "working_point")
            for method in methods}

        for i, wp in enumerate(wps):
            print(f"Plotting fixedWP SF comparisons for {wp} WP.")
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))
            ax.plot([pt_range[0], pt_range[-1]], [1, 1], c="black", lw=1)

            hdls = []
            ls = []

            for method in methods:
                c = f"{tagger}_{method}"
                if not c in list(cset): continue
                if not wp in available_wps[method]: continue
                corr = cset[c]

                color, style, label, _ = get_labels(method)
                if json_file == "ctagging":
                    sf = corr.evaluate("central", method, wp, flav, 0., bin_c)
                    try:
                        sf_up = corr.evaluate("up", method, wp, flav, 0., bin_c)
                        sf_dn = corr.evaluate("down", method, wp, flav, 0., bin_c)
                    except:
                        sf_up = sf_dn = sf
                else:
                    sf = corr.evaluate("central", wp, flav, 0., bin_c)
                    try:
                        sf_up = corr.evaluate("up", wp, flav, 0., bin_c)
                        sf_dn = corr.evaluate("down", wp, flav, 0., bin_c)
                    except:
                        sf_up = sf_dn = sf

                if binned:
                    all_sf = np.array([sf, sf_dn, sf_up])
                    sf_max = np.amax(all_sf, axis=0)
                    sf_min = np.amin(all_sf, axis=0)
                    # move the bin centers a bit
                    offset = i*(-1)**i
                    
                    hdl = ax.errorbar(bin_c+offset, sf,
                        yerr=(sf-sf_min, sf_max-sf),
                        xerr=(bin_w+offset, bin_w-offset),
                        ls="", ms=9, mew=2, lw=2,
                        marker=m, color=c)
                else:
                    hdl, = ax.plot(bin_c, sf, lw=3, marker="", ls=style, color=color)
                    sys_hdl = ax.fill_between(bin_c, sf_dn, sf_up, alpha=0.3, color=color)

                hdls.append(hdl)
                ls.append(f"{flav_text} jet SF ({label})")

            # labels and legends
            tagger_label = get_tagger_label(tagger)
            l2 = ax.legend(hdls, ls, loc="upper left", ncol=2,
                title=f"{tagger_label} ({wp} WP)", title_fontsize="small")
            plt.gca().add_artist(l2)

            ax.set_ylabel(f"$SF_{{{flav_text}}}$")
            ax.set_xlabel("jet $p_{T}$ (GeV)")

            mplhep.cms.label(ax=ax, data=True, label="Preliminary", rlabel=lumi_label)

            # make pretty
            ax.set_xscale("log")
            ax.set_xlim(pt_range[0], pt_range[-1])
            r = ax.get_ylim()[1]-ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0]+r*1.5)
            fig.tight_layout()
            plt.xticks(pt_bins, pt_bins)
            plt.minorticks_off()
            plt.grid(True, axis="x")

            # save
            out_file = os.path.join(plot_dir, f"{tagger}_comparison_{wp}WP.pdf")
            fig.savefig(out_file)
            fig.savefig(out_file.replace(".pdf",".png"))
            plt.close()




# go through database to generate plots
for filename in db:
    json_file = os.path.join(in_dir, f"{filename}_v{{v}}.json")
    i = -1
    while os.path.exists(json_file.format(v=i+1)):
        i += 1
    if i < 0:
        continue

    json_file = json_file.format(v=i)
    print(f"\n### generating plots for '{filename}' (v{i}) in campaign '{campaign}' ###\n")
    plot_configs = db[filename]
    
    cset = correctionlib.CorrectionSet.from_file(json_file)
    corrections = list(cset)
    
    # get the taggers
    taggers = list(set([c.split("_")[0] for c in corrections]))
    print(f"Found the following taggers: {taggers}")

    # loop over plot styles
    for plot in plot_configs:
        cfg = plot_configs[plot]

        # check if plot can be made if input is defined
        input_cset = cfg.get("input", None)
        available_taggers = taggers
        if input_cset:  
            if type(input_cset) == str:
                input_cset = [input_cset]
            
            do_plot = True
            available_taggers = []
            for t in taggers:
                available = True
                for c in input_cset:
                    if not c.format(tagger=t) in corrections:
                        available = False
                if available:
                    available_taggers.append(t)

            if len(available_taggers) == 0:
                do_plot = False

            if not do_plot: continue

        plot_dir = os.path.join(out_dir, plot)
        init_dir(plot_dir, do_php=eval(cfg.get("php", "True")))

        fnc = cfg["function"]
        cfg["in_json"] = json_file
        cfg["cset"] = cset
        cfg["taggers"] = available_taggers
        cfg["plot_dir"] = plot_dir
        eval(fnc)(**cfg)

        # delete empty directories
        delete_empty_dir(plot_dir)

# delete campaign directory if it is empty
delete_empty_dir(out_dir)
