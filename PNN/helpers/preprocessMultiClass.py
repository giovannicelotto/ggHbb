def preprocessMultiClass(dfs):
    '''
    dfs is a list of dataframes
    '''

    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    dfs_new = []
    for idx, df in enumerate(dfs):
        #df = df[df.leptonClass == leptonClass]

        
        #df = df[(df.jet1_pt>20) & (df.jet2_pt>20)]
        #df = df[(df.jet2_mass>0)] 
        #df = df[(df.jet1_mass>0)]
        #if 'jet3_mass' in df.columns:
        #    df = df[(df.jet3_mass>0)]
        # useless
        #df = df[(df.jet1_eta<2.5) & (df.jet1_eta>-2.5)]
        #df = df[(df.jet2_eta<2.5) & (df.jet2_eta>-2.5)]
        # end useless
        
        beforeCutMass = len(df)
        df = df[(df.dijet_mass>40) & (df.dijet_mass<300)]
        afterCutMass = len(df)
        print("Df Idx %d : Eff. cut mass %.1f"%(idx, afterCutMass/beforeCutMass*100))
        

        if df.isna().sum().sum()>0:
            print("Nan values : %d process %d "%(df.isna().sum().sum(), idx))
        #print("Filling jet1 qgl with 0. %d" %(df.jet1_qgl.isna().sum()))
        #print("Filling jet2 qgl with 0. %d" %(df.jet2_qgl.isna().sum()),"\n")

        #df.jet1_qgl = df.jet1_qgl.fillna(0.)
        #df.jet2_qgl = df.jet2_qgl.fillna(0.)
        #if 'jet3_qgl' in df.columns:
        #    df.jet3_qgl = df.jet3_qgl.fillna(0.)
        try:
            assert df.isna().sum().sum()==0
            assert df.isna().sum().sum()==0
        except:
            columns_with_nan = df.columns[df.isna().any()].tolist()
            # Find rows with NaN values
            rows_with_nan = df[df.isnull().any(axis=1)]
            print("Columns with NaN values:", columns_with_nan)
            print("\nRows with NaN values:")
            print(rows_with_nan)
            
        dfs_new.append(df)
    return dfs_new

