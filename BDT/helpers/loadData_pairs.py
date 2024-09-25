import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/BDT/helpers")
from load_dfPairs_sameLen import load_dfPairs_sameLen
from sklearn.model_selection import train_test_split

def loadData_pairs(path):
    dfs, min_len = load_dfPairs_sameLen(path)

    df = pd.concat(dfs)
    # Assuming pairs_df is the DataFrame you created
    X = df.drop(columns=['is_true_pair', 'massHypo'])  # Features
    y = df['is_true_pair']  # Labels

    ## Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)
    return X_train, X_test, y_train, y_test