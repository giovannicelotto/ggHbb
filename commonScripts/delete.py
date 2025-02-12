from functions import loadMultiParquet_Data_new
dfs, lumi_tot = loadMultiParquet_Data_new([0], nReals = 1, columns=['dijet_mass'], selectFileNumberList=[[1001, 989, 987]], returnFileNumberList=False)
print([len(df) for df in dfs])
print(lumi_tot)
