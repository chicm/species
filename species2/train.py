import settings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image

from utils import save_array, load_array, save_weights, load_best_weights, w_files_training
from utils import create_dense161, create_dense201, create_res101, create_res152, create_dense169, create_res50
from utils import create_vgg19bn, create_vgg16bn, create_dense121, create_inceptionv3
from utils import create_model
from cscreendataset import get_train_loader, get_val_loader

data_dir = settings.DATA_DIR

RESULT_DIR = data_dir + '/results_512'
MODEL_DIR = settings.MODEL_DIR
#batch_size = 4
epochs = 70

def train_model(model, criterion, optimizer, lr_scheduler, max_num = 2, init_lr=0.001, num_epochs=70):
    data_loaders = { 'train': get_train_loader(model), 'valid': get_val_loader(model)} 
    
    since = time.time()
    best_model = model
    best_acc = 0.0
    print(model.name)
    for epoch in range(num_epochs):
        epoch_since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch, init_lr=init_lr)
                model.train(True) 
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0
            for i, data in enumerate(data_loaders[phase]):
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                #optimizer.zero_grad()
                outputs = model(inputs)
                #preds = torch.sigmoid(outputs.data)
                preds = torch.ge(outputs.data, 0.5)
                #print("preds size:{}".format(preds.size()))
                #print("label size:{}".format(labels.data.size()))
                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    if (i+1) % 8 == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.int() == labels.data.int())
            epoch_loss = running_loss / data_loaders[phase].num
            epoch_acc = running_corrects / data_loaders[phase].num

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                save_weights(epoch_acc, model, epoch, max_num=max_num)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                #torch.save(best_model.state_dict(), w_file)
        print('epoch {}: {:.0f}s'.format(epoch, time.time()-epoch_since))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(w_files_training)
    return best_model

def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.6**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        print('existing lr = {}'.format(param_group['lr']))
        param_group['lr'] = lr
    return optimizer  

def cyc_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=6):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        lr = lr * 0.6
    if lr < 5e-6:
        lr = 0.0001
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer    

def train(model, init_lr = 0.001, freeze=False, num_epochs = epochs):
    criterion = nn.BCELoss()
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    #optimizer_ft = optim.Adam(model.parameters(), lr=init_lr)

    if freeze:
        print('training only classifier')
        if hasattr(model, 'fc'):
            optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
        elif hasattr(model, 'classifier'):
            optimizer_ft = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
        else:
            print('ERROR, no fc or classifier')
            exit()
    else:
        print('training full net')
        optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    model = train_model(model, criterion, optimizer_ft, cyc_lr_scheduler, init_lr=init_lr, 
                        num_epochs=num_epochs, max_num = model.max_num)
    return model


def train_net(model_name, freeze=False, num_epochs=epochs):
    print('Training {}...'.format(model_name))
    model = create_model(model_name)
    try:
        load_best_weights(model)
    except:
        print('Failed to load weigths')
    if not hasattr(model, 'max_num'):
        model.max_num = 2
    train(model, freeze=freeze, num_epochs=num_epochs)

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=1, help="train model")
parser.add_argument("--freeze", action='store_true', help="freeze conv layers")

args = parser.parse_args()
if args.train:
    print('start training model')
    mname = args.train[0]
    #train_net(mname)
    if args.freeze:
        train_net(mname, freeze=True, num_epochs=5)
    else:
        train_net(mname, freeze=False, num_epochs=epochs)

    print('done')
