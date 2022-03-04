import cv2
import numpy as np
import random
import math

white = (255,255,255)
black = (0,0,0)
angle_list = [math.pi * i / 12 for i in range(2, 11)] # 0 < angle < pi
thickness_list = [30, 40]
length_list = [100, 200]

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

def GenerateImage(type, angle_list = angle_list, thickness_list = thickness_list, length_list = length_list):
    start_point = [300, 600]
    end_point = [300, 300]
    direction_before = [0, -1]
    first_points = np.vstack([start_point, end_point])
    for first_angle in angle_list:
        if type == 1: # number of corner == 1
            start_point = first_points[-1]
            end_point_list, _ = FindEndPoint(start_point, direction_before, first_angle, length = 1000)
            for i in range(2):
                end_point = end_point_list[i]
                points = np.vstack([first_points, end_point])
                for thickness in thickness_list:
                    image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
                    cv2.polylines(img = image, pts = [points], isClosed = False, color = white, thickness = thickness)
                    if check_contours(np.copy(image)) == True and check_edge(image) == True:
                        image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    else:
                        image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('result', image)
                    cv2.waitKey(0)
        else:
            for first_length in length_list:
                start_point = first_points[-1]
                direction_before = [0, -1]
                first_end_point_list, first_direction_list = FindEndPoint(start_point, direction_before, first_angle, first_length)
                for i in range(2):
                    end_point = first_end_point_list[i]
                    direction_before = first_direction_list[i]
                    second_points = np.vstack([first_points, end_point])
                    start_point = second_points[-1]
                    for second_angle in angle_list:
                        if type == 2: # number of corner == 2
                            start_point = second_points[-1]
                            direction_before = first_direction_list[i]
                            end_point_list, _ = FindEndPoint(start_point, direction_before, second_angle, length = 1000)
                            for j in range(2):
                                end_point = end_point_list[j]
                                points = np.vstack([second_points, end_point])
                                for thickness in thickness_list:
                                    image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
                                    cv2.polylines(img = image, pts = [points], isClosed = False, color = white, thickness = thickness)
                                    if check_contours(np.copy(image)) == True and check_edge(image) == True:
                                        image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                    else:
                                        image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                    cv2.imshow('result', image)
                                    cv2.waitKey(0)
                        else:
                            for second_length in length_list:
                                start_point = second_points[-1]
                                direction_before = first_direction_list[i]
                                second_end_point_list, second_direction_list = FindEndPoint(start_point, direction_before, second_angle, second_length)
                                for j in range(2): # number of corner == 3
                                    end_point = second_end_point_list[j]
                                    direction_before = second_direction_list[j]
                                    third_points = np.vstack([second_points, end_point])
                                    start_point = third_points[-1]
                                    for third_angle in angle_list:
                                        end_point_list, _ = FindEndPoint(start_point, direction_before, third_angle, length = 1000)
                                        for k in range(2):
                                            end_point = end_point_list[k]
                                            points = np.vstack([third_points, end_point])
                                            for thickness in thickness_list:
                                                image = np.zeros(shape = [600, 600, 3], dtype = 'uint8')
                                                cv2.polylines(img = image, pts = [points], isClosed = False, color = white, thickness = thickness)
                                                if check_contours(np.copy(image)) == True and check_edge(image) == True:
                                                    image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                                else:
                                                    image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                                cv2.imshow('result', image)
                                                cv2.waitKey(0)
GenerateImage(type = 2)
# for i in range(1, 4):
#     GenerateImage(type = i)