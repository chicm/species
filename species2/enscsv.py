import pandas as pd 
import numpy as np

#src_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25.csv'
src_csvs = ['/home/chicm/ml/kgdata/species/results_tta/99484/tta15.csv', 
            '/home/chicm/ml/kgdata/species/results_512/420_1.csv',
	    '/home/chicm/ml/kgdata/species/results_512_2/420_2.csv',
	    '/home/chicm/ml/kgdata/species/results_224/224.csv',
	    '/home/chicm/ml/kgdata/species/results_380/size_380.csv',
	    '/home/chicm/ml/kgdata/species/results_256/256.csv',
            '/home/chicm/ml/kgdata/species/results_224_2/224_2.csv']
#tgt_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25_bin.csv'
tgt_csv = '/home/chicm/ml/kgdata/species/ensemble/ensemble7.csv'

dfvalue = []

for csv in src_csvs:
    df = pd.read_csv(csv)
    dfvalue.append(df.values[:,1])
    
mean_values = np.mean(dfvalue, axis=0)

df2 = pd.read_csv(src_csvs[0])
df2['invasive'] = mean_values
df2.to_csv(tgt_csv, index=False)
