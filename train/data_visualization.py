from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import csv
import cv2
import os
import numpy as np
import tqdm
import time
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.models import Model
from model import select_cnn_model

v1_source_label = './automatic_labelling_result/v1_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-18 09:10:19.496511/result/label.csv'
v2_source_label = './automatic_labelling_result/v2_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-17 16:34:36.396432/result/label.csv'
train_label_key = 'gt'

model = DenseNet121(weights='imagenet', include_top=True)
model = Model(model.input, model.layers[-2].output)

# for calculation time test
# ['MobileNetV2', 'EfficientNetB0', 'ResNet50']
# model, input_shape = select_cnn_model('ResNet50')

with open(v1_source_label, 'r') as f:
    label_reader = csv.DictReader(f)
    v1_source_label_list = list(label_reader)

with open(v2_source_label, 'r') as f:
    label_reader = csv.DictReader(f)
    v2_source_label_list = list(label_reader)

v1_label_list = []
v2_label_list = []
name_list = []
feature_list=[]

accum_time = 0
for element in tqdm.tqdm(v1_source_label_list):
    image = cv2.imread(element['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    start = time.time()
    feature = model.predict(np.array([cv2.resize(image, (224, 224))]))[0]
    accum_time += time.time() - start
    v1_label_list.append(int(element['gt']))
    name_list.append(os.path.splitext(os.path.basename(element['image_path']))[0])
    feature_list.append(feature)
print(accum_time, len(v1_source_label_list), accum_time/len(v1_source_label_list))

for element in v2_source_label_list:
    v2_label_list.append(int(element['gt']))



# Data Distribution by Road Width
v1_width_information={
    '111': [0,0],
    '114': [0,0],
    '117': [0,0]
}

v2_width_information={
    '111': [0,0],
    '114': [0,0],
    '117': [0,0]
}
for index, image_name in enumerate(name_list):
    width = image_name.split('_')[1]
    v1_width_information[width][v1_label_list[index]] += 1
    v2_width_information[width][v2_label_list[index]] += 1

print(v1_width_information)
print(v2_width_information)


categories = ['','','Feasibility False','', '','','Feasibility True','']
width = 0.25
fig, (ax1,ax2) = plt.subplots(1,2)
plt.tight_layout()

X = np.arange(2)
bar = ax1.bar(X+0.00, np.array(v1_width_information['111']), width, label='3.7 m')
for b in bar:
    height = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)
bar = ax1.bar(X+0.25, np.array(v1_width_information['114']), width, label='3.8 m')
for b in bar:
    height = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)
bar = ax1.bar(X+0.50, np.array(v1_width_information['117']), width, label='3.9 m')
for b in bar:
    height = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)

ax1.set_xticklabels(labels=categories, fontsize=25)
ax1.yaxis.set_tick_params(labelsize=25)
ax1.set_ylim([0,699])


ax1.set_ylabel('Number of Images', fontsize=25)
ax1.set_title('(a) V1 Data Distribution with Road Width', fontsize=30)
ax1.legend(loc='upper left', prop={'size': 25})


bar = ax2.bar(X+0.00, np.array(v2_width_information['111']), width, label='3.7 m')
for b in bar:
    height = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)
bar = ax2.bar(X+0.25, np.array(v2_width_information['114']), width, label='3.8 m')
for b in bar:
    height = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)
bar = ax2.bar(X+0.50, np.array(v2_width_information['117']), width, label='3.9 m')
for b in bar:
    height = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, height, height, ha='center', va='bottom', size = 20)

ax2.set_xticklabels(labels=categories, fontsize=25)
ax2.yaxis.set_tick_params(labelsize=25)
ax2.set_ylim([0,699])

ax2.set_ylabel('Number of Images', fontsize=25)
ax2.set_title('(b) V2 Data Distribution with Road Width', fontsize=30)
ax2.legend(loc='upper left', prop={'size': 25})

plt.show()
plt.cla()


# TSNE & PCA
tsne = TSNE(n_components=2)
tsne_scatter = tsne.fit_transform(feature_list)
pca = PCA(n_components=2)
pca_scatter = pca.fit_transform(feature_list)


tsne_x = np.array([x[0] for x in tsne_scatter])
tsne_y = np.array([x[1] for x in tsne_scatter])
pca_x = np.array([x[0] for x in pca_scatter])
pca_y = np.array([x[1] for x in pca_scatter])

v1_true_index = []
v1_false_index = []
for index, label in enumerate(v1_label_list):
    if label == 0:
        v1_false_index.append(index)
    else:
        v1_true_index.append(index)

v2_true_index = []
v2_false_index = []
for index, label in enumerate(v2_label_list):
    if label == 0:
        v2_false_index.append(index)
    else:
        v2_true_index.append(index)

s=5
alpha=0.5

plt.subplot(221)
plt.scatter(tsne_x[v1_true_index], tsne_y[v1_true_index], c=['blue']*len(v1_true_index), s=s, alpha=alpha, label='Feasibility True')
plt.scatter(tsne_x[v1_false_index], tsne_y[v1_false_index], c=['red']*len(v1_false_index), s=s, alpha=alpha, label='Feasibility False')
plt.title('(a) V1 t-SNE')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.legend(loc='upper right', prop={'size': 7})

plt.subplot(223)
plt.scatter(tsne_x[v2_true_index], tsne_y[v2_true_index], c=['blue']*len(v2_true_index), s=s, alpha=alpha, label='Feasibility True')
plt.scatter(tsne_x[v2_false_index], tsne_y[v2_false_index], c=['red']*len(v2_false_index), s=s, alpha=alpha, label='Feasibility False')
plt.title('(c) V2 t-SNE')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.legend(loc='upper right', prop={'size': 7})

plt.subplot(222)
plt.scatter(pca_x[v1_true_index], pca_y[v1_true_index], c=['blue']*len(v1_true_index), s=s, alpha=alpha, label='Feasibility True')
plt.scatter(pca_x[v1_false_index], pca_y[v1_false_index], c=['red']*len(v1_false_index), s=s, alpha=alpha, label='Feasibility False')
plt.title('(b) V1 PCA')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.legend(loc='upper right', prop={'size': 7})


plt.subplot(224)
plt.scatter(pca_x[v2_true_index], pca_y[v2_true_index], c=['blue']*len(v2_true_index), s=s, alpha=alpha, label='Feasibility True')
plt.scatter(pca_x[v2_false_index], pca_y[v2_false_index], c=['red']*len(v2_false_index), s=s, alpha=alpha, label='Feasibility False')
plt.title('(d) V2 PCA')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.legend(loc='upper right', prop={'size': 7})

plt.tight_layout()
plt.show()