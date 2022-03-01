import cv2
import numpy as np
import random
import math

white = (255,255,255)
black = (0,0,0)
def random_sign():
    if random.random() < 0.5:
        return 1
    else:
        return -1

def check_edge(image):
    edge = image[0,:,0]
    edge = np.append(edge, image[:,599,0])
    edge = np.append(edge, image[599,:,0][::-1])
    edge = np.append(edge, image[:,0,0][::-1])
    prev = edge[0]
    cluster = 0
    for pt in edge:
        if pt==0 and prev !=0:
            cluster+=1
        prev = pt
    if cluster != 2:
        return False
    else:
        return True


def get_random_endpoint(start_point, pre_direction, length=None):
    direction_x = random.uniform(-1,1)
    direction_y = random_sign() * math.sqrt((1-direction_x**2))
    direction = (direction_x, direction_y)
    dot_product = np.dot(pre_direction, direction)
    angle = np.arccos(dot_product)
    while not (30 < math.degrees(angle) < 120):
        direction_x = random.uniform(-1,1)
        direction_y = random_sign() * math.sqrt((1-direction_x**2))
        direction = (direction_x, direction_y)
        dot_product = np.dot(pre_direction, direction)
        angle = np.arccos(dot_product)
    if length is None:
        length = random.randint(50,400)
    end_point = np.array([int(start_point[0] + direction[0]*length), int(start_point[1] + direction[1]*length)])
    return end_point, direction

def generate_random_image():
    image = np.zeros(shape=[600, 600, 3], dtype=np.uint8)
    thickness = random.randint(120,150)
    start_point = (300,600)
    direction = (0,-1)
    length = 250
    end_point = np.array([int(start_point[0] + direction[0]*length), int(start_point[1] + direction[1]*length)])
    pts = np.array([start_point, end_point])
    for i in range(5):
        start_point = end_point
        end_point, direction = get_random_endpoint(start_point, direction)
        pts = np.vstack([pts,end_point])
        if (end_point[0]<0 or end_point[0]>600) or (end_point[1]<0 or end_point[1]>600):
            break
    else:
        start_point = end_point
        end_point, direction = get_random_endpoint(start_point, direction, length=500)
        pts = np.vstack([pts,end_point])
    cv2.polylines(img=image, pts=[pts], isClosed=False, color=white, thickness=thickness)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not (len(contours) == 1 and check_edge(np.copy(image))):
        image = generate_random_image()
    return image

for i in range(50):
    image = generate_random_image()
    cv2.imshow('image', image)
    cv2.waitKey(0)
