import cv2
import os
import numpy as np
import random
import math
import tqdm
import datetime

white = (255,255,255)
black = (0,0,0)

def random_sign():
    if random.random() < 0.5:
        return 1
    else:
        return -1

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
    prev = edge[0]
    # count the number of edge white cluster
    cluster = 0
    start_idx = 0
    first_start_idx = 0
    for idx, pt in enumerate(edge):
        if pt != 0 and prev == 0:
            start_idx = idx
            if first_start_idx == 0:
                first_start_idx = idx
        if pt == 0 and prev !=0:
            if idx - start_idx > 300 or idx - start_idx < 100:
                return False
            cluster+=1
        prev = pt

    if edge[-1] != 0:
        if edge[0] != 0:
            idx = len(edge) - 1 + first_start_idx
        else:
            idx = len(edge) - 1
        if idx - start_idx > 300 or idx - start_idx < 100:
            return False

    # num of cluster must be 2
    if cluster == 2:
        return True
    else:
        return False


def get_random_endpoint(start_point, pre_direction, length=None):
    direction_x = random.uniform(-1,1)
    direction_y = random_sign() * math.sqrt((1-direction_x**2))
    direction = (direction_x, direction_y)
    dot_product = np.dot(pre_direction, direction)
    angle = np.arccos(dot_product)
    # degree between 30 and 120
    while not (30 < math.degrees(angle) < 120):
        direction_x = random.uniform(-1,1)
        direction_y = random_sign() * math.sqrt((1-direction_x**2))
        direction = (direction_x, direction_y)
        dot_product = np.dot(pre_direction, direction)
        angle = np.arccos(dot_product)
    if length is None:
        # random line length
        length = random.randint(50,300)
    end_point = np.array([int(start_point[0] + direction[0]*length), int(start_point[1] + direction[1]*length)])
    return end_point, direction

def generate_random_image(thickness=None):
    image = np.zeros(shape=[600, 600, 3], dtype=np.uint8)
    if thickness is None:
        thickness = random.randint(120,140)
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
        # make definately finish by set length=1000
        start_point = end_point
        end_point, direction = get_random_endpoint(start_point, direction, length=1000)
        pts = np.vstack([pts,end_point])
    for_check_contours = np.copy(image)
    cv2.polylines(img=image, pts=[pts], isClosed=False, color=white, thickness=thickness)
    cv2.polylines(img=for_check_contours, pts=[pts], isClosed=False, color=white, thickness=5)
    # cv2.imshow('before_validation', cv2.resize(np.hstack([for_check_contours,image]),(600,300)))
    # cv2.waitKey(0)
    if not (check_contours(for_check_contours) and check_contours(np.copy(image)) and check_edge(np.copy(image))):
        image = generate_random_image(thickness)
    return image

if __name__=='__main__':
    save_path = 'generation_result'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, str(datetime.datetime.now()).replace(' ','_'))
    os.makedirs(save_path, exist_ok=True)
    for i in range(3):
        thickness = int(120 + i*10)
        print(thickness)
        for j in tqdm.tqdm(range(1000)):
            image = generate_random_image(thickness)
            image_path = os.path.join(save_path, f'{thickness}_{str(j).zfill(4)}.png')
            label_path = os.path.join(save_path, 'label_before_manual_labelling.txt')
            cv2.imwrite(image_path, image)
            with open(label_path, 'a') as f:
                f.write(f'{image_path} 300 600 \n')
            # cv2.imshow('result', image)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     cv2.destroyAllWindows()
            #     break