from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import LSTM
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from keras.callbacks import Callback
import numpy as np
import os
import time
import csv
import cv2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# Training parameters
batch_size = 16  # orig paper trained all networks with batch_size=128
epochs = 200
ratio_0=0.5
ratio_1=0.5
class_weight={
    0:ratio_0,
    1:ratio_1
}
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3
#only version 1
version=1
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the MIMIC_preprocessed data.
train_label="scherule_trainlabels.txt"
traindata_path="./trainlabels/"+train_label

f = open(traindata_path,"r")
traindata = f.readlines()
x_train=[]
y_train=[]
for element in traindata:
    image=cv2.imread(element.split(" ")[0])
    print(image.shape)
    x_train+=[image]
    y_train+=[np.array([int(element.split(" ")[1])])]
f.close()
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_train)
y_test=np.array(y_train)

#x_train=np.dstack((x_train[:,:,:18,:],x_train[:,:,24:,:]))
#x_test=np.dstack((x_test[:,:,:18,:],x_test[:,:,24:,:]))
num_of_features=15
f = open("lowlr_30_"+str(num_of_features)+"_"+str(time.time())+'_output_2c_redun_'+str(ratio_0)+':'+str(ratio_1)+'_.csv', 'w+')
csvwriter=csv.writer(f)
csvwriter.writerow(['accuracy','loss','precision','recall','f1_score','AUC','confusion matrix'])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# Input image dimensions.
input_shape = x_train.shape[1:]
print(input_shape)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, 2)
#y_test = keras.utils.to_categorical(y_test, 2)

class SkMetrics(Callback):        
    def on_epoch_end(self, epoch, logs={}):
        global f
        global csvwriter
        loss=self.model.evaluate(self.validation_data[0],self.validation_data[1])[0]
        yhat_probs = self.model.predict(self.validation_data[0])
        yhat_probs = [x[0] for x in yhat_probs]
        yhat_classes = [0 if x<0.5 else 1 for x in yhat_probs]
        testy = self.validation_data[1]
        print(yhat_classes[:30])
        print([x[0] for x in self.validation_data[1]][:30])
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(testy, yhat_classes)
        print('Accuracy: %f' % accuracy)
        print('Loss: %f' % loss)
        # precision tp / (tp + fp)
        precision = precision_score(testy, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(testy, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy, yhat_classes)
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(testy, yhat_classes)
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        auc = roc_auc_score(testy, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(testy, yhat_classes)
        print(matrix)
        csvwriter.writerow([accuracy,loss,precision,recall,f1,auc,str(matrix)])
def lr_schedule(epoch):
    lr = 1e-3
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


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_lstm(input_shape, depth):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=3)(x)
    y = Flatten()(x)

    
    #From here, lstm congegate
    """
    y = keras.layers.Reshape((1,256))(y)
    y = LSTM(64, activation='tanh', recurrent_activation='tanh', 
            use_bias=True, kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', bias_initializer='zeros', 
            dropout=0.1, recurrent_dropout=0.1, implementation=1)(y)
    y = keras.layers.Reshape((1,64))(y)
    
    y = LSTM(32, activation='tanh', recurrent_activation='tanh', 
            use_bias=True, kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', bias_initializer='zeros', 
            dropout=0.1, recurrent_dropout=0.1, implementation=1)(y)
    #y = keras.layers.Reshape((1,48))(y)   
    """
    """
    y = LSTM(24, activation='tanh', recurrent_activation='tanh', 
            use_bias=True, kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', bias_initializer='zeros', 
            dropout=0.1, recurrent_dropout=0.1, implementation=1)(y)
    y = keras.layers.Reshape((1,24))(y)
    y = LSTM(24, activation='tanh', recurrent_activation='tanh', 
            use_bias=True, kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', bias_initializer='zeros', 
            dropout=0.1, recurrent_dropout=0.1, implementation=1)(y)
    """
    #define output 1
    outputs = Dense(1, activation='sigmoid')(y)
    
    #define output one hot label
    #outputs = Dense(2, activation='softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = resnet_lstm(input_shape=input_shape, depth=depth)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
skmetrics=SkMetrics()
callbacks = [checkpoint, lr_reducer, lr_scheduler, skmetrics]



# Run training, with or without data augmentation.
print('Not using data augmentation.')
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            class_weight=class_weight,
            callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


f.close()    