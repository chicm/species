import os
import numpy as np
import glob
import shutil

import settings

from skimage import io, transform

CUR_DIR = os.getcwd()
print('CUR DIR:' + CUR_DIR)

blacklist = ['Type_2/2845.jpg', 'Type_2/5892.jpg', 'Type_1/5893.jpg',
             'Type_1/1339.jpg', 'Type_1/3068.jpg', 'Type_2/7.jpg',
             'Type_1/746.jpg', 'Type_1/2030.jpg', 'Type_1/4065.jpg',
             'Type_1/4702.jpg', 'Type_1/4706.jpg', 'Type_2/1813.jpg', 'Type_2/3086.jpg']
files_0522 = ['/Type_2/80.jpg', '/Type_3/968.jpg', '/Type_3/1120.jpg']


def create_directories():
    try_mkdir(settings.TEST_RESIZED_DIR)
    try_mkdir(settings.TRAIN_RESIZED_DIR)
    try_mkdir(settings.MODEL_DIR)
    try_mkdir(settings.VALID_DIR)


def try_mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass


def crop_from_center(img, width, height):
    y = img.shape[0]
    x = img.shape[1]
    offset_x = x // 2 - (width // 2)
    offset_y = y // 2 - (height // 2)
    return img[offset_y:offset_y + height, offset_x:offset_x + width]


def resize_640(src_dir, tgt_dir, match_str, target_size=640):
    files = glob.glob(src_dir + os.sep + match_str)
    print(src_dir + os.sep + match_str)
    for img_path in files:
        parts = img_path.split(os.sep)
        tgt_path = tgt_dir + os.sep + parts[len(parts) - 1]

        img = io.imread(img_path)
        if img.shape[0] > img.shape[1]:
            tile_size = (
                int(img.shape[0] * target_size / img.shape[1]), target_size)
        else:
            tile_size = (
                target_size, int(img.shape[1] * target_size / img.shape[0]))
        resized_image = transform.resize(img, tile_size, preserve_range=True)
        cropped_image = crop_from_center(resized_image, target_size, target_size)

        print(cropped_image[:5])

        io.imsave(tgt_path, cropped_image)


def resize_images():
    resize_640(settings.TRAIN_DIR, settings.TRAIN_RESIZED_DIR, '*.jpg')
    resize_640(settings.TEST_DIR, settings.TEST_RESIZED_DIR, '*.jpg')


def create_validation_data():
    files = glob.glob(settings.TRAIN_RESIZED_DIR + os.sep + '*/*.jpg')
    files = np.random.permutation(files)

    for i in range(600):
        fn = files[i]
        shutil.move(settings.TRAIN_RESIZED_DIR + '/' + fn, settings.VALID_DIR + '/' + fn)


def check():
    if not os.path.exists(settings.TRAIN_DIR):
        print('{} not found, please configure settings.INPUT_PATH correctly'.format(settings.TRAIN_DIR))
        return False
    if not os.path.exists(settings.TEST_DIR):
        print('{} not found, please configure settings.INPUT_PATH correctly'.format(settings.TEST_DIR))
        return False
    return True


if __name__ == "__main__":
    if not check():
        exit()

    if True:
        print('creating directories')
        create_directories()
        print('done')
    if True:
        print('creating resized images, this will take a while...')
        resize_images()
        print('done')
    if True:
        print('creating validation data')
        try:
            create_validation_data()
        except:
            pass
        print('done')
