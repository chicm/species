import argparse

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from cscreendataset import get_train_loader, get_val_loader
from settings import *
from utils import create_model
from utils import save_weights, load_best_weights, w_files_training

RESULT_DIR = DATA_DIR + '/results'

batch_size = 8
epochs = 100


def train_model(model, criterion, optimizer, lr_scheduler, max_num=2,
                init_lr=0.001, num_epochs=100):
    data_loaders = {'train': get_train_loader(model, batch_size=BATCH_SIZE),
                    'valid': get_val_loader(model, batch_size=BATCH_SIZE)}

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
            for data in data_loaders[phase]:
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(
                    labels.cuda())
                optimizer.zero_grad()
                outputs = model(inputs)
                #preds = torch.sigmoid(outputs.data)
                #print("preds size:{}".format(preds.size()))
                #print("label size:{}".format(labels.data.size()))
                preds = torch.ge(outputs.data, 0.5).view(labels.data.size())
                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.int() == labels.data.int())
            print("running_loss %d" % running_loss)
            print("running_corrects %d" % running_corrects)
            print("data_num %d" % data_loaders[phase].num)
            epoch_loss = running_loss / data_loaders[phase].num
            epoch_acc = running_corrects / data_loaders[phase].num

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                save_weights(epoch_acc, model, epoch, max_num=max_num)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model = copy.deepcopy(model)
                # torch.save(best_model.state_dict(), w_file)
        print('epoch {}: {:.0f}s'.format(epoch, time.time() - epoch_since))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(w_files_training)
    return best_model


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.6 ** (epoch // lr_decay_epoch))

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


def train(model, init_lr=0.001, num_epochs=epochs):
    criterion = nn.BCELoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    # optimizer_ft = optim.Adam(model.parameters(), lr=init_lr)

    model = train_model(model, criterion, optimizer_ft, cyc_lr_scheduler,
                        init_lr=init_lr,
                        num_epochs=num_epochs, max_num=model.max_num)
    return model


def train_net(model_name):
    print('Training {}...'.format(model_name))
    model = create_model(model_name)
    try:
        load_best_weights(model)
    except:
        print('Failed to load weigths')
    if not hasattr(model, 'max_num'):
        model.max_num = 2
    train(model)


parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=1, help="train model")

args = parser.parse_args()
if args.train:
    print('start training model')
    mname = args.train[0]
    train_net(mname)

    print('done')
