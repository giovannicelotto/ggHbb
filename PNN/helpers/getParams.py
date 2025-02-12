import os
def getParams(file_path="/t3home/gcelotto/ggHbb/PNN/helpers/params.txt"):
    hp = {
    #'nReal'             : 100,
    #'nMC'               : -1,
    'epochs'            : 5000,
    'patienceES'        : 100,
    'size'              : int(1e9),
    'val_split'         : 0.25,
    'test_split'        : 0.1,
    'learning_rate'     : 5e-4,
    'nNodes'            : [128, 64, 32],
    'batch_size'        : 25000,
    'lambda_dcor'        : 1,
    }

    # Determine the maximum length of the keys for alignment
    max_key_length = max(len(key) for key in hp.keys())
    
    # Print the keys and values in an aligned format
    for key, value in hp.items():  # You can directly iterate through items in a dictionary
        print(f"{key:<{max_key_length}} \t: {value}")
    print("-"*50)
    return hp