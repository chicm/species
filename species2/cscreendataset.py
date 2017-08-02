import os
import random

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import settings


def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class PlanetDataset(data.Dataset):
    def __init__(self, file_list_path, train_data=True, has_label=True,
                 transform=None, split=0.8):
        df_train = pd.read_csv(file_list_path)
        df_value = df_train.values
        df_value = np.random.permutation(df_value)
        if has_label:
            split_index = int(df_value.shape[0] * split)
            if train_data:
                split_data = df_value[:split_index]
            else:
                split_data = df_value[split_index:]
            # print(split_data.shape)
            file_names = [None] * split_data.shape[0]
            labels = [None] * split_data.shape[0]

            for index, line in enumerate(split_data):
                f, invasive = line
                file_names[index] = os.path.join(settings.TRAIN_DIR, str(f) + '.jpg')
                labels[index] = invasive
        else:
            file_names = [None] * df_train.values.shape[0]
            for index, line in enumerate(df_train.values):
                f, invasive = line
                file_names[index] = settings.TEST_DIR + '/' + str(int(f)) + '.jpg'
                # print(filenames[:100])
        self.transform = transform
        self.num = len(file_names)
        self.file_names = file_names
        self.train_data = train_data
        self.has_label = has_label

        self.images = []

        print("pre-reading images from files.")
        for file_name in tqdm.tqdm(file_names):
            self.images.append(pil_load(file_name))

        print("load %d images." % len(self.images))
        if has_label:
            self.labels = np.array(labels, dtype=np.float32)
            # print(self.labels.shape)

            # print(self.num)

    def __getitem__(self, index):
        # img = pil_load(self.file_names[index])
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.has_label:
            label = self.labels[index]
            return img, label, self.file_names[index]
        else:
            return img, self.file_names[index]

    def __len__(self):
        return self.num


def randomRotate(img):
    d = random.uniform(0, 360)
    img2 = img.rotate(d, resample=Image.NEAREST)
    return img2


def randomMaxScreen(img):
    if img.size[0] == img.size[1]:
        return img
    elif img.size[0] > img.size[1]:
        x1 = random.randint(0, img.size[0] - img.size[1])
        y1 = 0
        return img.crop((x1, y1, x1 + img.size[1], y1 + img.size[1]))
    elif img.size[0] < img.size[1]:
        x1 = 0
        y1 = random.randint(0, img.size[1] - img.size[0])
        return img.crop((x1, y1, x1 + img.size[0], y1 + img.size[0]))


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: randomMaxScreen(x)),
        transforms.Scale(317),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'trainv3': transforms.Compose([
        transforms.Lambda(lambda x: randomMaxScreen(x)),
        transforms.Scale(423),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validv3': transforms.Compose([
        transforms.Scale(301),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Scale(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testv3': transforms.Compose([
        transforms.Scale(301),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

'''
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'valid']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
dset_classes = dsets['train'].classes
save_array(CLASSES_FILE, dset_classes)
'''


def get_train_loader(model, batch_size=16, shuffle=True):
    if model.name.startswith('inception'):
        transkey = 'trainv3'
    else:
        transkey = 'train'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    # train_v2.csv
    print("train batch_size %d " % batch_size)
    dset = PlanetDataset(settings.DATA_DIR + os.sep + 'train_labels.csv',
                         transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle)
    dloader.num = dset.num
    return dloader


def get_val_loader(model, batch_size=16, shuffle=True):
    if model.name.startswith('inception'):
        transkey = 'validv3'
    else:
        transkey = 'valid'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    # train_v2.csv

    dset = PlanetDataset(settings.DATA_DIR + os.sep + 'train_labels.csv', train_data=False,
                         transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle)
    dloader.num = dset.num
    return dloader


def get_test_loader(model, batch_size=16, shuffle=False):
    if model.name.startswith('inception'):
        transkey = 'testv3'
    else:
        transkey = 'test'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size

    dset = PlanetDataset(settings.DATA_DIR + os.sep + 'sample_submission.csv', has_label=False,
                         transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle)
    dloader.num = dset.num
    return dloader


if __name__ == '__main__':
    loader = get_train_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
    loader = get_val_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
    loader = get_test_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
