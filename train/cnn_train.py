from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import numpy as np
import os
import csv
import datetime
from cnn_utils import lr_schedule, DataGenerator
from model import select_cnn_model, test_load_cnn_models

model_list = [
    'VGG16',
    'VGG19',  # too large memory
    'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'ResNet50', 'ResNet101', 'ResNet152',
    'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
    'InceptionV3', 'InceptionResNetV2',
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
    'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
    'Xception', 'NASNetLarge', 'NASNetMobile'
]

vehicle_name = 'Sherule'
save_dir = 'cnn_result_model'
# Training parameters
batch_size = 8
epochs = 50
test_model_list = False
train_label = 'automatic_labelling_result/Scherule_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-02-27 19:51:12.694159/result/label.csv'
test_label = 'data/test/testlabels/Scherule_testlabels.txt'
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.join(save_dir, vehicle_name)
os.makedirs(save_dir, exist_ok=True)
timestamp = str(datetime.datetime.now())
save_dir = os.path.join(save_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

with open(train_label, 'r') as f:
    label_reader = csv.DictReader(f)
    train_label_list = list(label_reader)

test_label_list = []
with open (test_label, 'r') as f:
    for line in f.readlines():
        image_path, _, _, gt = line.split()
        test_label_list.append({'image_path': image_path, 'gt': gt})


if test_model_list:
    test_load_cnn_models(model_list)

for model_name in model_list:
    csv_file_path = os.path.join(save_dir, f'{model_name}.csv')
    result_h5_filepath = os.path.join(save_dir, f'{model_name}.h5')
    model, input_shape = select_cnn_model(model_name)


    trainGen = DataGenerator(train_label_list, batch_size, input_shape[:2], label_key='gt')
    testGen = DataGenerator(test_label_list, batch_size, input_shape[:2])

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=result_h5_filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    csv_logger = CSVLogger(csv_file_path, separator=',', append=False)
    callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]

    try:
        print(f'{model_name} Train Start')
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=lr_schedule(0)),
                      metrics=['accuracy', Precision(), Recall(), AUC()])

        model.fit(trainGen,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=testGen,
                  shuffle=True,
                  callbacks=callbacks)

        # Test trained model.
        scores = model.evaluate(testGen, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    except Exception as e:
        print(f'{model_name} Fail to Train')
        pass
