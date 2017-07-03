import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import settings
from cscreendataset import get_test_loader
from utils import create_model
from utils import save_array, load_array

data_dir = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR

RESULT_DIR = data_dir + '/results'
PRED_FILE = RESULT_DIR + '/pred_ens.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth', 'dense169*pth', 'dense121*pth', 'inceptionv3*pth',
                  'res50*pth', 'res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']


def make_preds(net):
    loader = get_test_loader(net)
    preds = []
    m = nn.Softmax()
    net.eval()
    for i, (img, _) in enumerate(loader, 0):
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        # pred = m(outputs).data.cpu().tolist()
        pred = outputs.data.cpu().tolist()
        for p in pred:
            preds.append(p)
    return preds


def ensemble():
    preds_raw = []

    for match_str in w_file_matcher:
        # print(match_str)
        os.chdir(MODEL_DIR)
        w_files = glob.glob(match_str)
        # print('cur:' + os.getcwd())
        for w_file in w_files:
            full_w_file = MODEL_DIR + '/' + w_file
            mname = w_file.split('_')[0]
            print(full_w_file)
            model = create_model(mname)
            model.load_state_dict(torch.load(full_w_file))

            pred = make_preds(model)
            pred = np.array(pred)
            # print(pred[:100])
            preds_raw.append(pred)
            del model

    save_array(PRED_FILE_RAW, preds_raw)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_FILE, preds)


def submit(filename):
    # filenames = [f.split('/')[-1] for f, i in dsets.imgs]
    # filenames = get_stage1_test_loader('res50').filenames
    preds = load_array(PRED_FILE)
    print(preds[:100])
    subm_name = RESULT_DIR + '/' + filename
    df = pd.read_csv(data_dir + '/sample_submission.csv')
    df['invasive'] = preds
    print(df.head())
    df.to_csv(subm_name, index=False)

    preds2 = (preds > 0.5).astype(np.int)
    df2 = pd.read_csv(data_dir + '/sample_submission.csv') 
    df2['invasive'] = preds2
    df2.to_csv(subm_name + '01', index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--ens", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=1, help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble()
    print('done')
if args.sub:
    print('generating submision file...')
    submit(args.sub[0])
    print('done')
    print('Please find submisson file at: {}'.format(RESULT_DIR + '/' + args.sub[0]))
