import os, shutil, glob, sys
import pandas as pd
import numpy as np
import cv2


def resize(src_dir, tgt_dir, tgt_size):
    os.chdir(src_dir)
    files = glob.glob('*.jpg')

    for f in files:
        fn = src_dir+'/'+f
        tgt_fn = tgt_dir+'/'+f
        print(fn)
        print(tgt_fn)
        img = cv2.imread(fn)
        res = cv2.resize(img, tgt_size)
        #region = res[50:450, 50:450]
        #gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tgt_fn, res)
        
resize('/home/chicm/ml/kgdata/species/train', '/home/chicm/ml/kgdata/species/train-224', (224,224))
resize('/home/chicm/ml/kgdata/species/test', '/home/chicm/ml/kgdata/species/test-224', (224,224))
