from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import math
class DataGenerator(Sequence):
    def __init__(self,
                 label_list: list,
                 batch_size: int,
                 image_size: tuple) -> None:
        self.label_list = label_list
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return math.ceil(len(self.label_list) / self.batch_size)

    def __getitem__(self, index):
        image_batch = self.label_list[index * self.batch_size:(index + 1) * self.batch_size]
        
        Xs, Ys = [], []
        for i in range(len(image_batch)):
            X = cv2.resize(cv2.imread(image_batch[i]['image_path']), self.image_size)
            Y = 1 if image_batch[i]['gt'] == '1' else 0
            Xs.append(X)
            Ys.append(Y)

        Xs = np.array(Xs)
        Ys = np.array(Ys)

        return Xs, Ys

def lr_schedule(epoch):
    lr = 1e-3*0.5
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 90:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
