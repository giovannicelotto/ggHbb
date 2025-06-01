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



def get_input_paths(dataTaking, mass_hypos=[50, 70, 100, 200, 300]):
    base = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    folder = "training2"
    
    paths = [
        f"{base}/Data{dataTaking}/{folder}",
        f"{base}/MC/MINLOGluGluHToBB/training"
    ]

    paths += [f"{base}/MC/GluGluH_M{m}_ToBB" for m in mass_hypos]
    print(mass_hypos)
    masses = [125] + mass_hypos
    return paths, np.array(masses)



def apply_kinematic_cuts(dfs, boosted):
    cut_ranges = {
        0: {'dijet_pt': (None, 100), 'dijet_mass': (50, 300)},
        1: {'dijet_pt': (100, 160), 'dijet_mass': (50, 300)},
        2: {'dijet_pt': (160, None), 'dijet_mass': (50, 300)},

    }

    if boosted in cut_ranges:
        pt_min, pt_max = cut_ranges[boosted]['dijet_pt']
        mass_min, mass_max = cut_ranges[boosted]['dijet_mass']
        dfs = cut(dfs, 'dijet_pt', pt_min, pt_max)
        dfs = cut(dfs, 'dijet_mass', mass_min, mass_max)

    # For conservative training
    #if boosted == 22:
    #    for idx, df in enumerate(dfs):
    #        dfs[idx] = df[~((df.jet1_btagDeepFlavB > 0.71) & (df.jet2_btagDeepFlavB > 0.71))]
        
    #dfs_bkg = [dfs[0]]
    dfs_sig = dfs[1:]
    dfs_sig = cut(dfs_sig,'dR_jet1_genQuark',None, 0.2 )
    dfs_sig = cut(dfs_sig,'dR_jet2_genQuark',None, 0.2 )
    dfs_sig = cut(dfs_sig,'dpT_jet1_genQuark',None, 0.5 )
    dfs_sig = cut(dfs_sig,'dpT_jet2_genQuark',None, 0.5 )

    return [dfs[0]] + dfs_sig


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
            dfs[idx] = dfs[idx][dfs[idx]['dijet_mass'] < mass_limits[mass]]
    return dfs


def add_mass_hypothesis(dfs, massHypothesis, features):
    if 'massHypo' in features:
        for i, df in enumerate(dfs):
            dfs[i]['massHypo'] = df['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])
    return dfs


def assign_labels_and_weights(dfs, massHypothesis, boosted, sampling):
    dfSig = pd.concat(dfs[1:])
    dfBkg = dfs[0]

    dfSig['genMass'] = np.concatenate([np.ones(len(dfs[i])) * massHypothesis[i-1] for i in range(1, len(dfs))])
    dfBkg['genMass'] = np.zeros(len(dfs[0]))

    dfSig['Y'] = 1
    dfBkg['Y'] = 0

    if sampling:
        dfSig = uniform_sample(dfSig, column='dijet_mass', num_bins=101)
        dfBkg = uniform_sample(dfBkg, column='dijet_mass', num_bins=101)

    dfSig['W'] = dfSig.sf *  dfSig.PU_SF *  dfSig.btag_central / ((dfSig.sf *  dfSig.PU_SF *  dfSig.btag_central).sum())
    dfBkg['W'] = np.ones(len(dfBkg)) / len(dfBkg)

    return dfSig, dfBkg


def loadData_sampling(nReal, nMC, columnsToRead, featuresForTraining, test_split, boosted=False, dataTaking='1A', sampling=True):
    mass_hypos =[50, 70, 100, 200, 300]
    paths, massHypothesis = get_input_paths(dataTaking, mass_hypos=mass_hypos)

    btagTight = False#boosted in [1, 2]

    dfs = []
    for path in paths:
        fileNames = glob.glob(path+"/*.parquet")
        eg = os.path.basename(fileNames[0])
        match = re.match(r'(.+)_([0-9]+)\.parquet', eg)
        process = match.group(1)

        if process[:4]=='Data':
            if nReal !=-1:
                fileNames = fileNames[:nReal]
            print("Opening %d files for %s"%(len(fileNames), process))
            columnsToRead_ = [f for f in columnsToRead if ('gen' not in f) & ('btag_central' not in f) ]
        else:
            if nMC !=-1:
                fileNames = fileNames[:nMC]
            print("Opening %d files for %s"%(len(fileNames), process))
            columnsToRead_ = columnsToRead.copy()

        
        df = pd.read_parquet(fileNames, columns=columnsToRead_, filters=getCommonFilters(btagTight=btagTight))
        dfs.append(df)

    dfs = apply_kinematic_cuts(dfs, boosted)
    mass_hypos = [125, 50, 70, 100, 200, 300]
    mass_limits = {50: 120, 70: 140, 100:150}

    dfs = filter_mass_windows(dfs, mass_hypos, mass_limits)
    dfs = add_mass_hypothesis(dfs, massHypothesis, featuresForTraining)

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