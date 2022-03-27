import pygame
import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.util import img_as_ubyte
from math import tan, radians, degrees
from pygame.math import Vector2
from copy import deepcopy
import operator


class utility():
    def __init__(self):
        self.imagefile = './sample_data/train0.png'
        self.image = cv2.imread(self.imagefile)
        self.edge = cv2.Laplacian(self.image, cv2.CV_8U)
        self.edge = cv2.cvtColor(self.edge, cv2.COLOR_BGR2GRAY)

    def manual_control(self, car):
        pressed = pygame.key.get_pressed()
        print(pressed)
        if pressed[pygame.K_UP]:
            car.velocity.y = car.car_velocity
        elif pressed[pygame.K_DOWN]:
            car.velocity.y = -car.car_velocity
        else:
            car.velocity.y = 0

        if pressed[pygame.K_RIGHT]:
            if car.vehicle == 'car':
                car.steering = 30
            elif car.vehicle == 'spmt':
                car.velocity.x = -car.car_velocity
        elif pressed[pygame.K_LEFT]:
            if car.vehicle == 'car':
                car.steering = -30
            elif car.vehicle == 'spmt':
                car.velocity.x = car.car_velocity
        else:
            if car.vehicle == 'car':
                car.steering = 0
            elif car.vehicle == 'spmt':
                car.velocity.x = 0

        if car.vehicle == 'spmt':
            if pressed[pygame.K_q]:
                car.steering = 1
            elif pressed[pygame.K_e]:
                car.steering = -1
            else:
                car.steering = 0
        car.steering = max(-car.max_steering,
                           min(car.steering, car.max_steering))

    def rotate(self, origin, point, angle):
        '''
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        '''
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def get_instant_image(self, position, angle, carwidth, carlength, image_size, image_resize):
        x1, x2, x3, x4, y1, y2, y3, y4 = self.find_carpoints(
            position, angle, carwidth, carlength)
        x1 = int(x1-(position[0]-300))
        x2 = int(x2-(position[0]-300))
        x3 = int(x3-(position[0]-300))
        x4 = int(x4-(position[0]-300))
        y1 = int(y1-(position[1]-300))
        y2 = int(y2-(position[1]-300))
        y3 = int(y3-(position[1]-300))
        y4 = int(y4-(position[1]-300))
        cloneimage = np.array(self.image)
        # draw car shape
        rect_points = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])
        matrix = np.float32([[1, 0, 300-int(position[0])],
                            [0, 1, 300-int(position[1])]])
        traslated = cv2.warpAffine(
            cloneimage, matrix, (600, 600), borderValue=(255, 255, 255))
        traslated = cv2.fillConvexPoly(traslated, rect_points, (0, 0, 255))
        matrix = cv2.getRotationMatrix2D((300, 300), -angle, 1)
        rotated = cv2.warpAffine(
            traslated, matrix, (600, 600), borderValue=(0, 0, 0))
        scope = rotated[300-int(image_size/2):300+int(image_size/2),
                        300-int(image_size/2):300+int(image_size/2)]
        scope = cv2.resize(scope, (image_resize, image_resize))
        scope = cv2.cvtColor(scope, cv2.COLOR_BGR2GRAY)
        cv2.imshow('scope_image', scope)
        cv2.waitKey(1) & 0xFF
        return scope

    def lidar_data_sort(self, position, angle, pointlist):
        coords = pointlist
        center = tuple(position)
        result = sorted(coords, key=lambda coord: (
            95-degrees(angle)-degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return result

    def lidar_sensor(self, position, angle, maxdistance=1000, num_of_points=12):
        angle = radians(angle)
        lineimage = np.zeros([600, 600])
        startpoint = (position[0]+maxdistance*math.cos(angle),
                      position[1]-maxdistance*math.sin(angle))
        validindex = []
        for i in range(num_of_points):
            newpoint = self.rotate((int(position[0]), int(
                position[1])), startpoint, radians(i*(-360/num_of_points)))
            for j in range(5):
                validindex += [[newpoint[1], newpoint[0]]]
            lineimage = cv2.line(lineimage, (int(position[0]), int(
                position[1])), (int(newpoint[0]), int(newpoint[1])), 255, 2)
        validindex = np.array(validindex)
        contactpoint = img_as_ubyte(np.multiply(self.edge, lineimage))
        validindex = np.vstack([validindex, np.argwhere(contactpoint)])
        clustering = DBSCAN(eps=3, min_samples=2).fit(validindex)
        cluster_dict = {}
        count = 0
        for element in clustering.labels_:
            if element not in cluster_dict:
                cluster_dict[element] = [validindex[count]]
            else:
                cluster_dict[element] += [validindex[count]]
            count += 1
        result = []
        for element in cluster_dict:
            array = np.array(cluster_dict[element])
            result += [[np.average(array[:, 1]), np.average(array[:, 0])]]
        C = position
        remove_list = []
        for A in result:
            for B in result:
                A = np.array(A)
                B = np.array(B)
                if np.array_equal(A, B) == False:
                    maxlength = max(np.linalg.norm(
                        A-B), np.linalg.norm(B-C), np.linalg.norm(C-A))
                    minheight = abs(A[0]*(B[1]-C[1])+B[0] *
                                    (C[1]-A[1])+C[0]*(A[1]-B[1]))//maxlength
                    if minheight < 5:
                        if abs(maxlength-np.linalg.norm(A-B)) > 1:
                            if np.linalg.norm(B-C) >= np.linalg.norm(A-C):
                                remove_list += [list(B)]
                            else:
                                remove_list += [list(A)]
        for element in remove_list:
            try:
                result.remove(element)
            except:
                pass
        result = self.lidar_data_sort(position, angle, result)
        while len(result) < num_of_points:
            result += [position]
        if len(result) > num_of_points:
            result = result[:num_of_points]
        return result

    def front_lidar(self, position, angle, carlength, maxdistance):
        num_of_points = 1
        angle = radians(angle)
        lineimage = np.zeros([600, 600])
        startpoint = (position[0]-carlength/2*math.sin(angle)-maxdistance*math.sin(
            angle), position[1]-carlength/2*math.cos(angle)-maxdistance*math.cos(angle))
        validindex = []
        for i in range(num_of_points):
            newpoint = self.rotate((int(position[0]), int(
                position[1])), startpoint, radians(i*(-360/num_of_points)))
            for j in range(5):
                validindex += [[newpoint[1], newpoint[0]]]
            lineimage = cv2.line(lineimage, (int(position[0]), int(
                position[1])), (int(newpoint[0]), int(newpoint[1])), 255, 2)
        validindex = np.array(validindex)
        contactpoint = img_as_ubyte(np.multiply(self.edge, lineimage))
        validindex = np.vstack([validindex, np.argwhere(contactpoint)])
        clustering = DBSCAN(eps=3, min_samples=2).fit(validindex)
        cluster_dict = {}
        count = 0
        for element in clustering.labels_:
            if element not in cluster_dict:
                cluster_dict[element] = [validindex[count]]
            else:
                cluster_dict[element] += [validindex[count]]
            count += 1
        result = []
        for element in cluster_dict:
            array = np.array(cluster_dict[element])
            result += [[np.average(array[:, 1]), np.average(array[:, 0])]]
        C = position
        remove_list = []
        for A in result:
            for B in result:
                A = np.array(A)
                B = np.array(B)
                if np.array_equal(A, B) == False:
                    maxlength = max(np.linalg.norm(
                        A-B), np.linalg.norm(B-C), np.linalg.norm(C-A))
                    minheight = abs(A[0]*(B[1]-C[1])+B[0] *
                                    (C[1]-A[1])+C[0]*(A[1]-B[1]))//maxlength
                    if minheight < 5:
                        if abs(maxlength-np.linalg.norm(A-B)) > 1:
                            if np.linalg.norm(B-C) >= np.linalg.norm(A-C):
                                remove_list += [list(B)]
                            else:
                                remove_list += [list(A)]
        for element in remove_list:
            try:
                result.remove(element)
            except:
                pass
        result = self.lidar_data_sort(position, angle, result)
        while len(result) < num_of_points:
            result += [position]
        if len(result) > num_of_points:
            result = result[:num_of_points]
        return result

    def check_collision(self, position, angle, carwidth, carlength):
        x1, x2, x3, x4, y1, y2, y3, y4 = self.find_carpoints(
            position, angle, carwidth, carlength)
        cloneimage = np.copy(self.image)
        # draw car rectangle
        cv2.line(cloneimage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255))
        cv2.line(cloneimage, (int(x1), int(y1)), (int(x3), int(y3)), (255, 255, 255))
        cv2.line(cloneimage, (int(x4), int(y4)), (int(x2), int(y2)), (255, 255, 255))
        cv2.line(cloneimage, (int(x4), int(y4)), (int(x3), int(y3)), (255, 255, 255))
        ii = np.copy(self.image)
        cv2.line(ii, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0))
        cv2.line(ii, (int(x1), int(y1)), (int(x3), int(y3)), (255, 255, 0))
        cv2.line(ii, (int(x4), int(y4)), (int(x2), int(y2)), (255, 255, 0))
        cv2.line(ii, (int(x4), int(y4)), (int(x3), int(y3)), (255, 255, 0))
        # check sameness
        if np.array_equal(self.image, cloneimage):
            is_collision = False
        else:
            img_diff = self.image - cloneimage
            diff = list(zip(*np.where(img_diff[:,:,0]!=0)))
            for point in diff:
                y, x = point
                cv2.line(ii, (x,y), (x,y), color = (0, 0, 255), thickness = 5)
                print(self.image[y ,x], cloneimage[y, x])
            is_collision = True
        cv2.imshow('clone', ii)
        return is_collision

    def find_carpoints(self, position, angle, carwidth, carlength):
        # convert to radian
        angle = radians(angle)
        # apply offset
        carwidth -= 0.5
        carlength -= 0.5
        # lefttop
        x1 = position[0]-carwidth/2*math.cos(angle)-carlength/2*math.sin(angle)
        y1 = position[1]+carwidth/2*math.sin(angle)-carlength/2*math.cos(angle)
        # righttop
        x2 = position[0]+carwidth/2*math.cos(angle)-carlength/2*math.sin(angle)
        y2 = position[1]-carwidth/2*math.sin(angle)-carlength/2*math.cos(angle)
        # leftbottom
        x3 = position[0]-carwidth/2*math.cos(angle)+carlength/2*math.sin(angle)
        y3 = position[1]+carwidth/2*math.sin(angle)+carlength/2*math.cos(angle)
        # rightbottom
        x4 = position[0]+carwidth/2*math.cos(angle)+carlength/2*math.sin(angle)
        y4 = position[1]-carwidth/2*math.sin(angle)+carlength/2*math.cos(angle)

        return x1, x2, x3, x4, y1, y2, y3, y4


class Vehicle:
    def __init__(self, x, y, angle, vehicle_type, vehicle_name, max_steering=30):
        # from front of car
        # car spec
        if vehicle_type == 'car':
            if vehicle_name == 'Mack_Trucks_TerraPro':
                self.car_steering = 39.16
                self.scale = 5
                self.carwidth = 7.950*self.scale
                self.carlength = 27.054*self.scale
                self.curb_to_curb_r = 24.000*self.scale
                self.frontwheel_ratio = 6.054/(self.carlength/self.scale)
                self.rearwheel_ratio = (6.054+13.083)/(self.carlength/self.scale)
            elif vehicle_name == 'Pantechnicon_Removals_Van':
                self.car_steering = 36.91
                self.scale = 5.6
                self.carwidth = 8.202*self.scale
                self.carlength = 36.089*self.scale
                self.curb_to_curb_r = 40.026*self.scale
                self.frontwheel_ratio = 4.921/(self.carlength/self.scale)
                self.rearwheel_ratio = (4.921+21.982)/(self.carlength/self.scale)

        elif vehicle_type == 'spmt':
            if vehicle_name == 'Scherule':
                self.scale = 9.4
                self.carwidth = 2.430*self.scale*4
                self.carlength = 7.330*self.scale*2
            elif vehicle_name == 'Kamag':
                self.scale = 9.4 
                self.carwidth = 2.430*self.scale*3
                self.carlength = 7.000*self.scale*3
            elif vehicle_name == 'Type_1':
                self.carwidth = 90
                self.carlength = 160
            elif vehicle_name == 'Type_2':
                self.carwidth = 90
                self.carlength = 180
            elif vehicle_name == 'Type_3':
                self.carwidth = 100
                self.carlength = 160
            elif vehicle_name == 'Type_4':
                self.carwidth = 100
                self.carlength = 180
        self.car_velocity = -50
        self.vehicle = vehicle_type
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.max_steering = max_steering
        self.steering = 0.0
        self.rotation = 0.0
        self.rearvalid = 0

    def update(self, dt, image, type, rear_count):
        util = utility()
        util.image = image
        if type == 'car':
            if self.steering:
                l_dot = self.carlength * \
                    (self.rearwheel_ratio-self.frontwheel_ratio)
                #24.5 , 12.1
                # angle : 36.91, 40.74
                # c-c radiaus : 40.026, 24.278
                turning_radius = abs(tan(radians(self.steering)))/tan(radians(self.steering))*(
                    math.sqrt(self.curb_to_curb_r*self.curb_to_curb_r-l_dot*l_dot)-self.carwidth)
                angular_velocity = self.velocity.y / turning_radius
            else:
                angular_velocity = 0
            turning_center_ratio = (
                self.rearwheel_ratio+self.frontwheel_ratio)/2
            turning_length = math.sqrt((self.carwidth/2)*(self.carwidth/2)+self.carlength*(
                self.rearwheel_ratio-turning_center_ratio)*self.carlength*(self.rearwheel_ratio-turning_center_ratio))
            turning_vel = angular_velocity*turning_length
            theta = 90-degrees(math.atan(self.carlength *
                               (self.rearwheel_ratio-turning_center_ratio)/(self.carwidth/2)))
            if self.steering >= 0:
                turning_velocity = Vector2(-turning_vel*math.cos(
                    radians(theta)), turning_vel*math.sin(radians(theta)))
            else:
                turning_velocity = Vector2(-turning_vel*math.cos(
                    radians(theta)), -turning_vel*math.sin(radians(theta)))
            newposition = self.position + \
                self.velocity.rotate(-self.angle) * dt + \
                turning_velocity.rotate(-self.angle) * dt
            newangle = self.angle + degrees(angular_velocity) * dt

        elif type == 'spmt':
            newposition = self.position + \
                self.velocity.rotate(-self.angle) * dt
            newangle = self.angle+self.steering
        check_collision = util.check_collision(
            newposition, newangle, self.carwidth, self.carlength)
        check_finish = (newposition[0] < -10 or newposition[0] > 610) or (newposition[1] < -10 or newposition[1] > 610)
        if check_collision:
            self.position = self.position
            self.angle = self.angle
            self.rearvalid = 1
            return 0, rear_count

        if check_finish:
            self.position = self.position
            self.angle = self.angle
            self.rearvalid = 0
            rear_count = 0
            return 2, rear_count
        else:
            self.position = newposition
            self.angle = newangle
            if rear_count < 20 and self.rearvalid == 1:
                rear_count += 1
                self.rearvalid = 1
            else:
                rear_count = 0
                self.rearvalid = 0
            return 1, rear_count

    def step(self, action):
        if self.vehicle == 'car':
            car_steering = self.car_steering
            if self.rearvalid == 0:
                if action == 0:
                    self.velocity.y = self.car_velocity
                    self.steering = 0
                elif action == 1:
                    self.velocity.y = self.car_velocity
                    self.steering = car_steering
                elif action == 2:
                    self.velocity.y = self.car_velocity
                    self.steering = -car_steering
                elif action == 3:
                    self.velocity.y = -self.car_velocity
                    self.steering = 0
                elif action == 4:
                    self.velocity.y = -self.car_velocity
                    self.steering = car_steering
                elif action == 5:
                    self.velocity.y = -self.car_velocity
                    self.steering = -car_steering
            if self.rearvalid == 1:
                if action == 0:
                    self.velocity.y = self.car_velocity
                    self.steering = 0
                elif action == 1:
                    self.velocity.y = self.car_velocity
                    self.steering = car_steering
                elif action == 2:
                    self.velocity.y = self.car_velocity
                    self.steering = -car_steering
                elif action == 3:
                    self.velocity.y = -self.car_velocity
                    self.steering = 0
                elif action == 4:
                    self.velocity.y = -self.car_velocity
                    self.steering = car_steering
                elif action == 5:
                    self.velocity.y = -self.car_velocity
                    self.steering = -car_steering
        elif self.vehicle == 'spmt':
            if action == 0:
                self.velocity.x = 0
                self.velocity.y = self.car_velocity
                self.steering = 0
            elif action == 1:
                self.velocity.x = -self.car_velocity
                self.velocity.y = self.car_velocity
                self.steering = 0
            elif action == 2:
                self.velocity.x = self.car_velocity
                self.velocity.y = self.car_velocity
                self.steering = 0
            elif action == 3:
                self.velocity.x = 0
                self.velocity.y = -self.car_velocity
                self.steering = 0
            elif action == 4:
                self.velocity.x = -self.car_velocity
                self.velocity.y = -self.car_velocity
                self.steering = 0
            elif action == 5:
                self.velocity.x = self.car_velocity
                self.velocity.y = -self.car_velocity
                self.steering = 0
            elif action == 6:
                self.velocity.x = 0
                self.velocity.y = 0
                self.steering = 1
            elif action == 7:
                self.velocity.x = 0
                self.velocity.y = 0
                self.steering = -1
            elif action == 8:
                self.velocity.x = -self.car_velocity
                self.velocity.y = 0
                self.steering = 0
            elif action == 9:
                self.velocity.x = self.car_velocity
                self.velocity.y = 0
                self.steering = 0
            elif action == 10:
                self.velocity.x = 0
                self.velocity.y = self.car_velocity
                self.steering = 1
            elif action == 11:
                self.velocity.x = 0
                self.velocity.y = self.car_velocity
                self.steering = -1
            elif action == 12:
                self.velocity.x = 0
                self.velocity.y = -self.car_velocity
                self.steering = 1
            elif action == 13:
                self.velocity.x = 0
                self.velocity.y = -self.car_velocity
                self.steering = -1
