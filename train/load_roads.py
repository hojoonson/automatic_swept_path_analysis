import json
import os
import cv2
import datetime
import numpy as np
from tqdm import tqdm

white = (255,255,255)
black = (0,0,0)
json_directory = '../draw/picture_parameter'
json_name = 'parameter_0.json'
json_path = os.path.join(json_directory, json_name)
imshow = False

def DrawRoadsUsingParameter(data):
    corner_points = np.array(data['corner_points'][i])
    curve_line_points = np.array(data['curve_line_points'][i])
    thickness = data['thickness'][i]
    image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
    cv2.polylines(img = image, pts = [curve_line_points], isClosed = False, color = white, thickness = thickness)
    marked_image = np.copy(image)
    for point in corner_points:
        cv2.line(marked_image, point, point, color = (0, 0, 255), thickness = 10)
    return image, marked_image

if __name__=='__main__':
    save_path = 'generation_result'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, str(datetime.datetime.now()).replace(' ','_'))
    os.makedirs(save_path, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    picture_num = len(data['corner_points'])
    for i in tqdm(range(picture_num)):
        image, marked_image = DrawRoadsUsingParameter(data)

        image_path = os.path.join(save_path, f'{str(i).zfill(4)}.png')
        marked_image_path = os.path.join(save_path, f'marked_{str(i).zfill(4)}.png')
        label_path = os.path.join(save_path, 'label_before_manual_labelling.txt')
        
        cv2.imwrite(image_path, image)
        cv2.imwrite(marked_image_path, marked_image)

        with open(label_path, 'a') as f:
            f.write(f'{image_path} 300 600 \n')

        if imshow:
            cv2.imshow('result', image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break