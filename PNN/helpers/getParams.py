def getParams():
    hp = {
    'epochs'            : 3000,
    'patienceES'        : 40,
    'validation_split'  : 0.2,
    'test_split'        : 0.2,
    'learning_rate'     : 5*1e-5,
    'nNodes'            : [64, 32, 16],
    'batch_size'        : 2048,
        }
    hp['nDense']=len(hp['nNodes'])
    return hp