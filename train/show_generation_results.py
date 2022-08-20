import cv2
import glob
import os
import numpy as np

image_path = './generation_result/total_test_data_0411'

image_files = glob.glob(os.path.join(image_path, '0*.png'))
image_files = image_files[30:len(image_files)-1:83]
print(len(image_files))
col = 6
row = 4

col_index = 1
row_index = 1
for image_file in image_files:
    image = cv2.imread(image_file)
    image = cv2.resize(image, (100,100))
    # top, bottom, left, right
    if col_index == 1:
        if row_index == row:
            image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(224,224,224))
        else:
            image = cv2.copyMakeBorder(image, 10, 0, 10, 10, cv2.BORDER_CONSTANT, value=(224,224,224))
    else:
        if row_index == row:
            image = cv2.copyMakeBorder(image, 10, 10, 0, 10, cv2.BORDER_CONSTANT, value=(224,224,224))
        else:
            image = cv2.copyMakeBorder(image, 10, 0, 0, 10, cv2.BORDER_CONSTANT, value=(224,224,224))
    

    if col_index == col:
        row_concat = np.hstack([row_concat, image])
        if row_index ==1:
            col_concat = np.copy(row_concat)
        else:
            col_concat = np.vstack([col_concat, row_concat])
        col_index = 1
        row_index +=1
    else:
        if col_index == 1:
            row_concat = image
        else:
            row_concat = np.hstack([row_concat, image])
        col_index +=1

cv2.imwrite('road_generation.png', col_concat)