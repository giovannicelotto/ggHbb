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
    # First, split the data into 60% training and 40% temp (test+validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1999)

    # Then, split the temp set into 50% validation and 50% test (i.e., 20% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1999)

    return X_train, X_val, X_test, y_train, y_val, y_test