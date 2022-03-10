import cv2
import numpy as np
import random
import math

white = (255,255,255)
black = (0,0,0)
angle_list = [math.pi * i / 12 for i in range(2, 11)] # 0 < angle < pi
thickness_list = [30, 40]
length_list = [50, 100]

def check_contours(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # num of contours must be 1
    if len(contours) == 1:
        return True
    else:
        return False

def check_edge(image):
    # append 4 serial edges
    edge = image[0,:,0]
    edge = np.append(edge, image[:,599,0])
    edge = np.append(edge, image[599,:,0][::-1])
    edge = np.append(edge, image[:,0,0][::-1])
 
    count = 0
    change_point = []
    for i in range(1, 2399):
        if edge[i - 1] == 0 and edge[i] == 255 and edge[i + 1] == 255:
            change_point.append(i)
            count += 1

    if count == 2:
        return True
    else:
        return False

def FindEndPoint(start_point, direction_before, angle, length):
    a = direction_before[0]
    b = direction_before[1]
    if b == 0:
        x1 = math.cos(angle)
        x2 = math.cos(angle)
        y1 = math.sin(angle)
        y2 = math.sin(angle)
    else:
        x1 = a * math.cos(angle) - abs(b * math.sin(angle))
        x2 = a * math.cos(angle) + abs(b * math.sin(angle))
        y1 = b * math.cos(angle) + a * abs(b * math.sin(angle)) / b
        y2 = b * math.cos(angle) - a * abs(b * math.sin(angle)) / b
    direction1 = [x1, y1]
    direction2 = [x2, y2]
    direction_list = [direction1, direction2]
    end_point1 = [int(start_point[0] + direction1[0] * length), int(start_point[1] + direction1[1] * length)]
    end_point2 = [int(start_point[0] + direction2[0] * length), int(start_point[1] + direction2[1] * length)]
    end_point_list = [end_point1, end_point2]
    return end_point_list, direction_list

def GenerateImage(corner_num, end_point_list = [[300, 300]], direction_before_list = [[0, -1]], index = 0, point_set_before = [300, 600], count = 0, angle_list = angle_list, thickness_list = thickness_list, length_list = length_list):
    end_point = end_point_list[index]
    direction_before = direction_before_list[index]
    point_set = np.vstack([point_set_before, end_point])
    start_point = point_set[-1]
    count += 1
    for angle in angle_list:
        if corner_num == count:
            end_point_list, _ = FindEndPoint(start_point, direction_before, angle, length = 1000)
            for i in range(2):
                end_point = end_point_list[i]
                new_point_set = np.vstack([point_set, end_point])
                for thickness in thickness_list:
                    image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
                    cv2.polylines(img = image, pts = [new_point_set], isClosed = False, color = white, thickness = thickness)
                    if check_contours(np.copy(image)) == True and check_edge(image) == True:
                        image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    else:
                        image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('result', image)
                    cv2.waitKey(0)
        else:
            for length in length_list:
                start_point = point_set[-1]
                direction_before = direction_before_list[index]
                end_point_list, direction_list = FindEndPoint(start_point, direction_before, angle, length)
                for i in range(2):
                    GenerateImage(corner_num, end_point_list, direction_list, i, point_set, count)

corner_num = 4
GenerateImage(corner_num)