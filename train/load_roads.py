import json
import os
import cv2
import numpy as np
from tqdm import tqdm

white = (255,255,255)
black = (0,0,0)
json_directory = '../draw/picture_parameter'
json_name = 'parameter_1.json'
json_path = os.path.join(json_directory, json_name)
imshow = False

def DrawRoadsUsingParameter(json_path, save_corner = True):
    with open(json_path, 'r') as f:
        data = json.load(f)

    picture_num = len(data['corner_points'])
    for i in tqdm(range(picture_num)):
        corner_points = np.array(data['corner_points'][i])
        curve_line_points = np.array(data['curve_line_points'][i])
        thickness = data['thickness'][i]
        image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
        cv2.polylines(img = image, pts = [curve_line_points], isClosed = False, color = white, thickness = thickness)
        if save_corner == True:
            for point in corner_points:
                cv2.line(image, point, point, color = (0, 0, 255), thickness = 10)
        if imshow:
            cv2.imshow('result', image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

DrawRoadsUsingParameter(json_path)