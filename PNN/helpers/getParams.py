def getParams():
    hp = {
    'nReal'             : 9,
    'nMC'               : -1,
    'epochs'            : 5000,
    'patienceES'        : 40,
    'val_split'         : 0.25,
    'test_split'        : 0.2,
    'learning_rate'     : 1e-5,
    'nNodes'            : [64, 32, 16],
    'batch_size'        : 512,
    'lambda_reg'        : 0.02,
        }
    # Determine the maximum length of the keys for alignment
    max_key_length = max(len(key) for key in hp.keys())

    # Print the keys and values in an aligned format
    for key, value in hp.items():  # You can directly iterate through items in a dictionary
        print(f"{key:<{max_key_length}} \t: {value}")
    print("-"*50)
    return hp