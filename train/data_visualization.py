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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

v1_source_label = './automatic_labelling_result/v1_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-18 09:10:19.496511/result/label.csv'
v2_source_label = './automatic_labelling_result/v2_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-17 16:34:36.396432/result/label.csv'
train_label_key = 'gt'

model = ResNet50(weights='imagenet', include_top=True)
model = Model(model.input, model.layers[-2].output)


with open(v1_source_label, 'r') as f:
    label_reader = csv.DictReader(f)
    v1_source_label_list = list(label_reader)

with open(v2_source_label, 'r') as f:
    label_reader = csv.DictReader(f)
    v2_source_label_list = list(label_reader)

image_list = []
v1_label_list = []
v2_label_list = []
name_list = []
feature_list=[]
for element in tqdm.tqdm(v1_source_label_list):
    image = cv2.imread(element['image_path'])
    # feature = model.predict(np.array([cv2.resize(image, (224, 224))]))[0]
    image_list.append(image)
    v1_label_list.append(int(element['gt']))
    name_list.append(os.path.splitext(os.path.basename(element['image_path']))[0])
    # feature_list.append(feature)
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


categories = ['Feasibility False', 'Feasibility True']
width = 0.8 
fig, (ax1,ax2) = plt.subplots(1,2)
plt.tight_layout()

ax1.bar(categories, np.array(v1_width_information['111']), width, label='111')
ax1.bar(categories, np.array(v1_width_information['114']), width, bottom=np.array(v1_width_information['111']), label='114')
ax1.bar(categories, np.array(v1_width_information['117']), width, bottom=np.array(v1_width_information['111'])+np.array(v1_width_information['114']), label='117')

ax1.set_ylabel('Number of Images')
ax1.set_title('(a) V1 Data Distribution with Road Width')
ax1.legend(loc='upper left')

ax2.bar(categories, np.array(v2_width_information['111']), width, label='111')
ax2.bar(categories, np.array(v2_width_information['114']), width, bottom=np.array(v2_width_information['111']), label='114')
ax2.bar(categories, np.array(v2_width_information['117']), width, bottom=np.array(v2_width_information['111'])+np.array(v2_width_information['114']), label='117')

ax2.set_ylabel('Number of Images')
ax2.set_title('(b) V2 Data Distribution with Road Width')
ax2.legend(loc='upper left')
plt.show()
plt.cla()


# TSNE
# tsne = TSNE(n_components=2)
# tsne_scatter = tsne.fit_transform(feature_list)
# pca = PCA(n_components=2)
# pca_scatter = pca.fit_transform(feature_list)


# tsne_x = np.array([x[0] for x in tsne_scatter])
# tsne_y = np.array([x[1] for x in tsne_scatter])
# pca_x = np.array([x[0] for x in pca_scatter])
# pca_y = np.array([x[1] for x in pca_scatter])

# v1_true_index = []
# v1_false_index = []
# for index, label in enumerate(v1_label_list):
#     if label == 0:
#         v1_false_index.append(index)
#     else:
#         v1_true_index.append(index)

# v2_true_index = []
# v2_false_index = []
# for index, label in enumerate(v2_label_list):
#     if label == 0:
#         v2_false_index.append(index)
#     else:
#         v2_true_index.append(index)

# s=5
# alpha=0.5

# plt.subplot(221)
# plt.scatter(tsne_x[v1_true_index], tsne_y[v1_true_index], c=['blue']*len(v1_true_index), s=s, alpha=alpha, label='Feasibility True')
# plt.scatter(tsne_x[v1_false_index], tsne_y[v1_false_index], c=['red']*len(v1_false_index), s=s, alpha=alpha, label='Feasibility False')
# plt.title('(a) V1 T-SNE')
# plt.xlabel('component-1')
# plt.ylabel('component-2')
# plt.legend(loc='lower right', prop={'size': 7})

# plt.subplot(222)
# plt.scatter(tsne_x[v2_true_index], tsne_y[v2_true_index], c=['blue']*len(v2_true_index), s=s, alpha=alpha, label='Feasibility True')
# plt.scatter(tsne_x[v2_false_index], tsne_y[v2_false_index], c=['red']*len(v2_false_index), s=s, alpha=alpha, label='Feasibility False')
# plt.title('(b) V2 T-SNE')
# plt.xlabel('component-1')
# plt.ylabel('component-2')
# plt.legend(loc='lower right', prop={'size': 7})

# plt.subplot(223)
# plt.scatter(pca_x[v1_true_index], pca_y[v1_true_index], c=['blue']*len(v1_true_index), s=s, alpha=alpha, label='Feasibility True')
# plt.scatter(pca_x[v1_false_index], pca_y[v1_false_index], c=['red']*len(v1_false_index), s=s, alpha=alpha, label='Feasibility False')
# plt.title('(c) V1 PCA')
# plt.xlabel('component-1')
# plt.ylabel('component-2')
# plt.legend(loc='lower right', prop={'size': 7})


# plt.subplot(224)
# plt.scatter(pca_x[v2_true_index], pca_y[v2_true_index], c=['blue']*len(v2_true_index), s=s, alpha=alpha, label='Feasibility True')
# plt.scatter(pca_x[v2_false_index], pca_y[v2_false_index], c=['red']*len(v2_false_index), s=s, alpha=alpha, label='Feasibility False')
# plt.title('(d) V2 PCA')
# plt.xlabel('component-1')
# plt.ylabel('component-2')
# plt.legend(loc='lower right', prop={'size': 7})

# plt.tight_layout()
# plt.show()