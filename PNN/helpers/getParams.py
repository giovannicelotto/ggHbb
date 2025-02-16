import os
def print_params(hp):
    """Print dictionary keys and values in an aligned format."""
    max_key_length = max(len(key) for key in hp.keys())
    
    for key, value in hp.items():
        print(f"{key:<{max_key_length}} \t: {value}")
    print("-" * 50)

def getParams():
    hp = {
    #'nReal'             : 100,
    #'nMC'               : -1,
    'epochs'            : 5000,
    'patienceES'        : 200,
    'size'              : int(1e9),
    'val_split'         : 0.1,
    'test_split'        : 0.05,
    'learning_rate'     : 5e-4,
    'min_learning_rate'     : 5e-6,
    'nNodes'            : [128, 64, 32],
    'batch_size'        : 25000,
    'lambda_dcor'        : 1,
    }
    print_params(hp)
    return hp