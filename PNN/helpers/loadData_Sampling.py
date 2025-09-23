import numpy as np
from functions import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def uniform_sample(df, column='dijet_mass', num_bins=20):
    """
    Sample rows from a DataFrame to ensure a uniform distribution in the specified column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column to sample uniformly.
    - num_bins (int): Number of bins to divide the column into.

    Returns:
    - pd.DataFrame: A subset of df with uniform distribution in the specified column.
    """

    # Create bins based on the target column
    df['bins'] = pd.cut(df[column], bins=num_bins)

    # Find the smallest bin size to ensure fair sampling
    min_bin_size = df['bins'].value_counts().min()

    # Sample an equal number of rows from each bin
    df_uniform = df.groupby('bins', group_keys=False, observed=True).apply(lambda x: x.sample(n=min_bin_size, replace=False))


    # Drop the temporary bin column
    df_uniform = df_uniform.drop(columns=['bins'], axis=1)

    return df_uniform



def get_input_paths(dataTaking, mass_hypos=[50, 70, 100, 200, 300], boosted=3):
    base = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    folder = "training"
    if boosted==4:
        input("These are old Paths. Press Any key to continue")
        paths = [
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB_o/Data1D/training",
        f"{base}/MC/MINLOGluGluHToBB/training"
    ]
    
    else:
        paths = [
        f"{base}/Data{dataTaking}/{folder}",
        f"{base}/MC/MINLOGluGluHToBB/training"
    ]
    if boosted==4:
        masses = [125]
    else:    
        paths += [f"{base}/MC/GluGluH_M{m}_ToBB" for m in mass_hypos]
        print(mass_hypos)
        masses = [125] + mass_hypos
    return paths, np.array(masses)



def apply_kinematic_cuts(dfs, boosted):
    cut_ranges = {
        0: {'dijet_pt': (None, 100), 'dijet_mass': (50, 300)},
        1: {'dijet_pt': (100, 160), 'dijet_mass': (50, 300)},
        2: {'dijet_pt': (160, None), 'dijet_mass': (50, 300)},
        3: {'dijet_pt': (100, None), 'dijet_mass': (50, 300)},
        4: {'dijet_pt': (50, 80), 'dijet_mass': (80, 170)},

    }

    if boosted in cut_ranges:
        pt_min, pt_max = cut_ranges[boosted]['dijet_pt']
        mass_min, mass_max = cut_ranges[boosted]['dijet_mass']
        print("pt min", pt_min)
        dfs = cut(dfs, 'dijet_pt', pt_min, pt_max)
        dfs = cut(dfs, 'dijet_mass', mass_min, mass_max)

    # For conservative training
    #if boosted == 22:
    #    for idx, df in enumerate(dfs):
    #        dfs[idx] = df[~((df.jet1_btagDeepFlavB > 0.71) & (df.jet2_btagDeepFlavB > 0.71))]
        
    #dfs_bkg = [dfs[0]]
        

    genMatched = False
    if genMatched:
        dfs_sig = dfs[1:]
        dfs_sig = cut(dfs_sig,'dR_jet1_genQuark',None, 0.2 )
        dfs_sig = cut(dfs_sig,'dR_jet2_genQuark',None, 0.2 )
        dfs_sig = cut(dfs_sig,'dpT_jet1_genQuark',None, 0.5 )
        dfs_sig = cut(dfs_sig,'dpT_jet2_genQuark',None, 0.5 )

        return [dfs[0]] + dfs_sig
    else:
        return dfs


def filter_mass_windows(dfs, mass_hypos, mass_limits):
    """
    Apply dijet_mass upper cuts to signal datasets based on mass hypotheses.

    Parameters:
    - dfs (list of pd.DataFrame): List of dataframes where dfs[0] is background and dfs[1:] are signals.
    - mass_hypos (list): List of mass hypotheses (e.g., [50, 70, 100, 200, 300]).
    - mass_limits (dict): Mapping of mass hypothesis to dijet_mass upper limit (e.g., {50: 120, 70: 140}).

    Returns:
    - list of pd.DataFrame: Filtered dataframes.
    """
    for i, mass in enumerate(mass_hypos):
        if mass in mass_limits:
            idx = i + 1  # offset because dfs[0] is background
            dfs[idx] = dfs[idx][dfs[idx]['dijet_mass'] < mass_limits[mass][1]]
            dfs[idx] = dfs[idx][dfs[idx]['dijet_mass'] > mass_limits[mass][0]]
    return dfs


def add_mass_hypothesis(dfs, massHypothesis, features):
    if 'massHypo' in features:
        for i, df in enumerate(dfs):
            dfs[i]['massHypo'] = df['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])
    return dfs


def assign_labels_and_weights(dfs, massHypothesis, boosted, sampling):
    if boosted!=0:
        dfSig = pd.concat(dfs[1:])
        dfBkg = dfs[0]

        dfSig['genMass'] = np.concatenate([np.ones(len(dfs[i])) * massHypothesis[i-1] for i in range(1, len(dfs))])
        dfBkg['genMass'] = np.zeros(len(dfs[0]))
    else:
        dfSig=dfs[1]
        dfBkg=dfs[0]

    dfSig['Y'] = 1
    dfBkg['Y'] = 0

    if sampling:
        dfSig = uniform_sample(dfSig, column='dijet_mass', num_bins=101)
        dfBkg = uniform_sample(dfBkg, column='dijet_mass', num_bins=101)

    dfSig['W'] = dfSig.sf *  dfSig.PU_SF *  dfSig.btag_central * abs(dfSig.genWeight) / ((dfSig.sf *  dfSig.PU_SF *  dfSig.btag_central * abs(dfSig.genWeight)).sum())
    dfBkg['W'] = np.ones(len(dfBkg)) / len(dfBkg)
    dfBkg["btag_central"] = 1
    dfBkg["genWeight"] = 1

    return dfSig, dfBkg


def loadData_sampling(nReal, nMC, columnsToRead, featuresForTraining, test_split, boosted=False, dataTaking='1A', sampling=True, btagTight=True, mass_hypos =[]):
    
    paths, massHypothesis = get_input_paths(dataTaking, mass_hypos=mass_hypos, boosted=boosted)

    #boosted in [1, 2]

    dfs = []
    for path in paths:
        fileNames = glob.glob(path+"/*.parquet", recursive=True)
        print(path+"/*.parquet")
        eg = os.path.basename(fileNames[0])
        match = re.match(r'(.+)_([0-9]+)\.parquet', eg)
        process = match.group(1)

        if process[:4]=='Data':
            if nReal !=-1:
                fileNames = fileNames[:nReal]
            print("[Data] Opening %d files for %s"%(len(fileNames), process))
            columnsToRead_ = [f for f in columnsToRead if (('gen' not in f) & ('btag_central' not in f) & ('sf' not in f))]

        else:
            if nMC !=-1:
                fileNames = fileNames[:nMC]
            print("[MC] Opening %d files for %s"%(len(fileNames), process))
            columnsToRead_ = columnsToRead.copy()

        
        df = pd.read_parquet(fileNames, columns=columnsToRead_, filters=getCommonFilters(btagTight=btagTight, cutDijet=True))
        if process[:4]=='Data':
            df['sf']=1
            df['btag_central']=1
            df['PU_SF']=1
        dfs.append(df)

    dfs = apply_kinematic_cuts(dfs, boosted)
    mass_hypos = [125] + mass_hypos
    mass_limits = {50: [0, 120], 70: [0,140], 100:[0,150], 300:[150,300]}

    print(mass_hypos)
    dfs = filter_mass_windows(dfs, mass_hypos, mass_limits)
    dfs = add_mass_hypothesis(dfs, massHypothesis, featuresForTraining)
# Here remove signal events wheter more
#
#
#   
    if boosted==0:
        print("WARNING cutting signal in order to have same number of events in every bin of dijet mass")
        dfSig = pd.concat(dfs[1:])
        dfBkg = dfs[0]
        dfSig['genMass'] = np.concatenate([np.ones(len(dfs[i])) * massHypothesis[i-1] for i in range(1, len(dfs))])
        dfBkg['genMass'] = np.zeros(len(dfs[0]))

        mass_bins = np.quantile(dfBkg.dijet_mass.values, np.linspace(0, 1, 15))
        mass_bins[0], mass_bins[-1] = 50., 300.
        print("This the mass_binning for ABCDisco")
        print(mass_bins)

        train_signal = dfSig.copy()
        train_signal['mass_bin'] = np.digitize(train_signal.dijet_mass.values, mass_bins) - 1

        # Step 3: Determine minimum count across bins
        min_count = train_signal['mass_bin'].value_counts().min()

        # Step 4: Downsample signal events in each bin
        dfSig = (
            train_signal.groupby('mass_bin')
            .apply(lambda df: df.sample(min_count, random_state=42))
            .reset_index(drop=True)
        )

        dfs = [dfBkg, dfSig]



# Finished
    dfSig, dfBkg = assign_labels_and_weights(dfs, massHypothesis, boosted, sampling=sampling)

    for m in massHypothesis:
        print(f"{len(dfSig[dfSig.genMass==m])} elements in df {m}")
    print(f"{len(dfBkg[dfBkg.genMass==0])} elements in df 0")

    # Merge, shuffle, and split
    common_columns = dfBkg.columns.intersection(dfSig.columns)
    dfBkg = dfBkg[common_columns]
    dfSig = dfSig[common_columns]
    X = pd.concat([dfSig, dfBkg])
    Y = X.pop('Y').values
    W = X.pop('W').values
    genMass = X.pop('genMass').values

    X, Y, W, genMass = shuffle(X, Y, W, genMass, random_state=1999)

    return train_test_split(X, Y, W, genMass, test_size=test_split, random_state=1999)