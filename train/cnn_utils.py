from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import math
import random

class DataGenerator(Sequence):
    def __init__(self,
                 label_list: list,
                 batch_size: int,
                 image_size: tuple,
                 label_key: str='gt') -> None:
        self.label_list = label_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_key = label_key
    def on_epoch_end(self):
        random.shuffle(self.label_list)

    def __len__(self):
        return math.ceil(len(self.label_list) / self.batch_size)

    def __getitem__(self, index):
        image_batch = self.label_list[index * self.batch_size:(index + 1) * self.batch_size]

        Xs, Ys = [], []
        for i in range(len(image_batch)):
            X = cv2.resize(cv2.imread(image_batch[i]['image_path']), self.image_size)
            Y = 1 if image_batch[i][self.label_key] == '1' else 0
            Xs.append(X)
            Ys.append(Y)

        Xs = np.array(Xs)
        Ys = np.array(Ys)

        return Xs, Ys


def lr_schedule(epoch):
    lr = 1e-3*0.5
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
