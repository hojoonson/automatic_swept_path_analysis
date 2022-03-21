import json
import os
import cv2
import numpy as np

white = (255,255,255)
black = (0,0,0)
json_directory = './draw/picture_parameter'
json_name = 'parameter_0.json'
json_path = os.path.join(json_directory, json_name)

def DrawRoadsUsingParameter(json_path, show_corner = True):
    with open(json_path, 'r') as f:
        data = json.load(f)

    picture_num = len(data['corner_points'])
    for i in range(picture_num):
        corner_points = np.array(data['corner_points'][i])
        curve_line_points = np.array(data['curve_line_points'][i])
        thickness = data['thickness'][i]
        image = image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
        cv2.polylines(img = image, pts = [curve_line_points], isClosed = False, color = white, thickness = thickness)
        if show_corner == True:
            for point in corner_points:
                cv2.line(image, point, point, color = (0, 0, 255), thickness = 10)
        cv2.imshow('result', image)
        cv2.waitKey(0)
        
DrawRoadsUsingParameter(json_path)