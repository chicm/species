import os

DATA_DIR = 'C:\\Users\\caitsithx\\dl_data\\invasive-species-monitoring'
TRAIN_DIR = DATA_DIR + os.sep + 'train'
TEST_DIR = DATA_DIR + os.sep + 'test'
VALID_DIR = DATA_DIR + os.sep + 'valid'
RESULT_DIR = DATA_DIR + os.sep + 'result'

TRAIN_RESIZED_DIR = DATA_DIR + os.sep + 'train-640'
TEST_RESIZED_DIR = DATA_DIR + os.sep + 'test-640'
MODEL_DIR = DATA_DIR + os.sep + 'models'
BATCH_SIZE = 24
