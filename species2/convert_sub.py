import pandas as pd 
import numpy as np

#src_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25.csv'
src_csv = '/home/chicm/ml/kgdata/species/results/sub8.csv'
#tgt_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25_bin.csv'
tgt_csv = '/home/chicm/ml/kgdata/species/results/sub8_bin.csv'

df1 = pd.read_csv(src_csv)
df2 = pd.read_csv(src_csv)

df2['invasive'] = (df1.values[:, 1] >= 0.5).astype(np.uint8)
df2.to_csv(tgt_csv, index=False)
