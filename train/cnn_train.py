from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.models import load_model
import numpy as np
import os
import csv
import datetime
import time
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

vehicle_name = 'v2'
with open(f'./data/cnn_data/{vehicle_name}/train.csv', 'r') as f:
    label_reader = csv.DictReader(f)
    train_images = list(label_reader)
    train_images = [os.path.basename(x['image_path']) for x in train_images]
with open(f'./data/cnn_data/{vehicle_name}/test.csv', 'r') as f:
    label_reader = csv.DictReader(f)
    test_images = list(label_reader)
    test_images = [os.path.basename(x['image_path']) for x in test_images]

for image in test_images:
    assert image not in train_images

save_dir = 'cnn_result_model'
# Training parameters
batch_size = 8
epochs = 100
test_model_list = False


# source_label = './automatic_labelling_result/v1_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-18 09:10:19.496511/result/label.csv'
source_label = './automatic_labelling_result/v2_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-17 16:34:36.396432/result/label.csv'
train_label_key = 'gt'

# test_label = 'data/test/testlabels/Scherule_testlabels.txt'
# test_label = 'data/test/testlabels/Kamag_testlabels.txt'

os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.join(save_dir, vehicle_name)
os.makedirs(save_dir, exist_ok=True)
timestamp = str(datetime.datetime.now())
save_dir = os.path.join(save_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

with open(source_label, 'r') as f:
    label_reader = csv.DictReader(f)
    source_label_list = list(label_reader)

train_label_list = [x for x in source_label_list if os.path.basename(x['image_path']) in train_images]
test_label_list = [x for x in source_label_list if os.path.basename(x['image_path']) in test_images]
    

if test_model_list:
    test_load_cnn_models(model_list)

result_summary=[]
for model_name in model_list:
    csv_file_path = os.path.join(save_dir, f'{model_name}.csv')
    result_h5_filepath = os.path.join(save_dir, f'{model_name}.h5')
    model, input_shape = select_cnn_model(model_name)


    trainGen = DataGenerator(train_label_list, batch_size, input_shape[:2], label_key=train_label_key)
    testGen = DataGenerator(test_label_list, batch_size, input_shape[:2])

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=result_h5_filepath,
                                 monitor='val_loss',
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
        best_model = load_model(result_h5_filepath)
        loss, accuracy, precision, recall, auc = best_model.evaluate(testGen, verbose=1)
        result_summary.append({
            'model': model_name,
            'loss': loss,
            'accuracy': accuracy, 
            'precision': precision,
            'recall': recall,
            'auc': auc
        })
        with open(os.path.join(save_dir, f'{vehicle_name}-{train_label_key}_summary.csv'), 'w') as result_file:
            dict_writer = csv.DictWriter(result_file, result_summary[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(result_summary)

    except Exception as e:
        print(e)
        print(f'{model_name} Fail to Train')
        pass
