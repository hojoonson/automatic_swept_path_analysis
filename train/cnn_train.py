from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import numpy as np
import os
import csv
import datetime
from train_utils import lr_schedule, SkMetrics
from model import select_cnn_model, test_load_cnn_models

# Prepare model model saving directory.
vehicle_name = 'Sherule'
save_dir = 'cnn_result_model'
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.join(save_dir, vehicle_name)
os.makedirs(save_dir, exist_ok=True)
model_list = [
    # 'VGG16', 'VGG19',
    'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
    # 'DenseNet121', 'DenseNet169', 'DenseNet201',
    # 'ResNet50', 'ResNet101', 'ResNet152',
    # 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
    # 'InceptionV3', 'InceptionResNetV2',
    # 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
    # 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
    # 'Xception', 'NASNetLarge', 'NASNetMobile'
]
test_load_cnn_models(model_list)
model_name = 'MobileNet'
label_path = ''
# timestamp = str(datetime.datetime.now()).replace(' ', '_')
timestamp = '1234'
csv_file_path = os.path.join(save_dir, f'{model_name}_{timestamp}.csv')
with open(csv_file_path, 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['accuracy', 'loss', 'precision', 'recall', 'f1_score', 'AUC', 'tn', 'fp', 'fn', 'tp'])
result_h5_filepath = os.path.join(save_dir, f'{model_name}_{timestamp}.h5')

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=result_h5_filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
skmetrics = SkMetrics()
callbacks = [checkpoint, lr_reducer, lr_scheduler, skmetrics]
model = select_cnn_model(model_name)
