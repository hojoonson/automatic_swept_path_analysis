import cv2
import numpy as np
import random
import math
from Bezier import Bezier
import json
import os

white = (255,255,255)
black = (0,0,0)
angle_list = [math.pi * i / 6 for i in range(1, 6)] # 0 < angle < pi
thickness_list = [40]
length_list = [200, 400]
map_size = 600
parameter = {}
parameter['corner_points'] = []
parameter['curve_line_points'] = []
parameter['thickness'] = []
save_directory = './draw/picture_parameter'
save_name = 'parameter'

def check_contours(pts):
    test_image = np.zeros(shape = [map_size, map_size, 3], dtype = 'uint8')
    pts = np.round(pts)
    pts = pts.astype(np.int32)
    cv2.polylines(img = test_image, pts = [pts], isClosed = False, color = white, thickness = 1)
    imgray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # num of contours must be 1
    if len(contours) == 1:
        return True
    else:
        return False

def check_edge(image):
    # append 4 serial edges
    edge = image[0,:,0]
    edge = np.append(edge, image[:, map_size - 1, 0])
    edge = np.append(edge, image[map_size - 1, :, 0][::-1])
    edge = np.append(edge, image[:, 0, 0][::-1])
 
    count = 0
    change_point = []
    for i in range(1, map_size * 4 - 1):
        if edge[i - 1] == 0 and edge[i] == 255 and edge[i + 1] == 255:
            change_point.append(i)
            count += 1

    if count == 2:
        return True
    else:
        return False

def GetY(x, x1, y1, x2, y2):
    if x1 == x2:
        return False
    else:
        return round((y2 -y1) * (x - x1) / (x2 - x1) + y1, 5)

def GetX(y, x1, y1, x2, y2):
    if y1 == y2:
        return False
    else:
        return round((x2 - x1) * (y - y1) / (y2 - y1) + x1, 5)

def CheckBoundaryIntersection(end_point):
    x2 = round(end_point[0], 5)
    y2 = round(end_point[1], 5)
    if x2 > 0 and x2 < map_size and y2 > 0 and y2 < map_size:
        return False
    else:
        return True

def FindIntersectionPoint(start_point, end_point):
    x1 = round(start_point[0], 5)
    y1 = round(start_point[1], 5)
    x2 = round(end_point[0], 5)
    y2 = round(end_point[1], 5)
    if CheckBoundaryIntersection(end_point) == False:
        raise Exception("no intersection")
    else:
        point_x0 = GetY(0, x1, y1, x2, y2)
        point_x600 = GetY(map_size, x1, y1, x2, y2)
        point_y0 = GetX(0, x1, y1, x2, y2)
        point_y600 = GetX(map_size, x1, y1, x2, y2)
        if (point_x0 >= 0 and point_x0 <= map_size and x2 <= 0) and ((y1 <= point_x0 and point_x0 <= y2) or (y2 <= point_x0 and point_x0 <= y1)): # intersect with x = 0
            return [0, point_x0]
        elif (point_x600 >= 0 and point_x600 <= map_size and x2 >= map_size) and ((y1 <= point_x600 and point_x600 <= y2) or (y2 <= point_x600 and point_x600 <= y1)): # intersect with x = 600
            return [map_size, point_x600]
        elif (point_y0 >= 0 and point_y0 <= map_size and y2 <= 0) and ((x1 <= point_y0 and point_y0 <= x2) or (x2 <= point_y0 and point_y0 <= x1)): # intersect with y = 0
            return [point_y0, 0]
        elif (point_y600 >= 0 and point_y600 <= map_size and y2 >= map_size) and ((x1 <= point_y600 and point_y600 <= x2) or (x2 <= point_y600 and point_y600 <= x1)): # intersect with y = 600
            return [point_y600, map_size]
        else:
            raise Exception("error")

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
    end_point1 = [start_point[0] + direction1[0] * length, start_point[1] + direction1[1] * length]
    end_point2 = [start_point[0] + direction2[0] * length, start_point[1] + direction2[1] * length]
    if CheckBoundaryIntersection(end_point1) == True:
        end_point1 = FindIntersectionPoint(start_point, end_point1)
    if CheckBoundaryIntersection(end_point2) == True:
        end_point2 = FindIntersectionPoint(start_point, end_point2)
    end_point_list = [end_point1, end_point2]
    return end_point_list, direction_list

def DrawStraightRoad(image, pts, thickness):
    cv2.polylines(img = image, pts = [pts], isClosed = False, color = white, thickness = thickness)
    if check_contours(pts) == True and check_edge(image) == True:
        image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)

def DrawCurvedRoad(image, pts, thickness):
    corner_num = pts.shape[0] - 2
    point_b = pts[0]
    for i in range(corner_num):
        first_point = point_b
        middle_point = pts[i + 1]
        last_point = pts[i + 2]
        ratio_a = random.uniform(0, 1)
        ratio_b = random.uniform(0, 0.5)
        point_a = [first_point[0] + ratio_a * (middle_point[0] - first_point[0]), first_point[1] + ratio_a * (middle_point[1] - first_point[1])]
        point_b = [middle_point[0] + ratio_b * (last_point[0] - middle_point[0]), middle_point[1] + ratio_b * (last_point[1] - middle_point[1])]
        t_points = np.arange(0, 1, 0.01)
        reference_points = np.vstack([point_a, middle_point, point_b])
        curve_points = Bezier.Curve(t_points, reference_points)
        if i == 0:
            point_set = np.vstack([first_point, curve_points])
        else:
            point_set = np.vstack([point_set, first_point, curve_points])

        if i == corner_num - 1:
            end_point = [2 * last_point[0] - middle_point[0], 2 * last_point[1] - middle_point[1]]
            point_set = np.vstack([point_set, last_point, end_point])
            point_set = np.round(point_set)
            point_set = point_set.astype(np.int32)
            cv2.polylines(img = image, pts = [point_set], isClosed = False, color = white, thickness = thickness)

    if check_contours(pts) == True and check_edge(image) == True:
        temp_pts = np.round(pts)
        temp_pts = temp_pts.astype(np.int32)
        parameter['corner_points'].append(temp_pts.tolist())
        parameter['curve_line_points'].append(point_set.tolist())
        parameter['thickness'].append(thickness)
        image = cv2.putText(image, "True", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "False", (10, 30), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)

    for point in pts: # mark a reference point
        point = point.astype(np.int32)
        cv2.line(image, point, point, color = (0, 0, 255), thickness = 10)

def GenerateImage(corner_num, end_point_list = [[map_size / 2, map_size / 2]], direction_before_list = [[0, -1]], index = 0, point_set_before = [map_size / 2, map_size], count = 0, angle_list = angle_list, thickness_list = thickness_list, length_list = length_list):
    end_point = end_point_list[index]
    direction_before = direction_before_list[index]
    point_set = np.vstack([point_set_before, end_point])
    start_point = point_set[-1]
    if CheckBoundaryIntersection(end_point) == False:
        count += 1
        for angle in angle_list:
            if corner_num == count:
                end_point_list, _ = FindEndPoint(start_point, direction_before, angle, length = map_size * 2)
                for i in range(2):
                    end_point = end_point_list[i]
                    new_point_set = np.vstack([point_set, end_point])
                    for thickness in thickness_list:
                        image = np.zeros(shape = [map_size, map_size, 3], dtype = 'uint8')
                        # DrawStraightRoad(image, new_point_set, thickness)
                        DrawCurvedRoad(image, new_point_set, thickness)
                        cv2.imshow('result', image)
                        cv2.waitKey(0)
            else:
                for length in length_list:
                    start_point = point_set[-1]
                    direction_before = direction_before_list[index]
                    end_point_list, direction_list = FindEndPoint(start_point, direction_before, angle, length)
                    for i in range(2):
                        GenerateImage(corner_num, end_point_list, direction_list, i, point_set, count)
    else:
        for thickness in thickness_list:
            image = np.zeros(shape = [map_size, map_size, 3], dtype = 'uint8')
            DrawCurvedRoad(image, point_set, thickness)
            # DrawStraightRoad(image, point_set, thickness)
            cv2.line(image, [1100, 1100], [1100, 1100], color = (0, 255, 0), thickness = 20)
            cv2.imshow('result', image)
            cv2.waitKey(0)

corner_num = 1
GenerateImage(corner_num)

if os.path.exists(save_directory) == False:
    os.makedirs(save_directory)

count = 0
save_path = os.path.join(save_directory, save_name + '_{}.json'.format(count))
while os.path.exists(save_path) == True:
    count += 1
    save_path = os.path.join(save_directory, save_name + '_{}.json'.format(count))

with open(save_path, 'w') as f:
    json.dump(parameter, f)