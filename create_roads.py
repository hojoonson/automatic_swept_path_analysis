import cv2
import numpy as np
import random
import math

white=(255,255,255)

def random_sign():
    if random.random() < 0.5:
        return 1
    else:
        return -1

def get_random_endpoint(start_point):
    direction_x = random.uniform(-1,1)
    direction_y = random_sign() * math.sqrt((1-direction_x**2))
    direction = (direction_x, direction_y)
    length = random.randint(50,400)
    end_point = int(start_point[0] + direction[0]*length), int(start_point[1] + direction[1]*length)
    return end_point

def generate_random_image():
    image = np.zeros(shape=[600, 600, 3], dtype=np.uint8)

    thickness = random.randint(100,120)
    start_point = (300,600)
    direction = (0,-1)
    length = 300
    start_point = int(start_point[0]), int(start_point[1])
    end_point = int(start_point[0] + direction[0]*length), int(start_point[1] + direction[1]*length)
    cv2.line(img=image, pt1=start_point, pt2=end_point, color=white, thickness=thickness)

    for i in range(20):
        if (end_point[0]<0 or end_point[0]>600) or (end_point[1]<0 or end_point[1]>600):
            break
        start_point = end_point
        # thickness = random.randint(100,120)
        while image[end_point[1], end_point[0]][0] == 255:
            end_point = get_random_endpoint(start_point)
            if (end_point[0]<0 or end_point[0]>600) or (end_point[1]<0 or end_point[1]>600):
                break
        cv2.line(img=image, pt1=start_point, pt2=end_point, color=white, thickness=thickness)
    return image

for i in range(20):
    image = generate_random_image()
    cv2.imshow('image', image)
    cv2.waitKey(0)
