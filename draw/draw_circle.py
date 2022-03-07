import os
import pygame
import math
import cv2
import time
import numpy as np
#from scipy.misc import bytescale
from sklearn.cluster import DBSCAN
from skimage import transform
from math import tan, radians, degrees, copysign
from pygame.math import Vector2
from functools import reduce
import operator

imagefile="ground.png"
image=cv2.imread(imagefile)
edge=cv2.Laplacian(image,cv2.CV_8U)
edge=cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)
class utility:
    def rotate(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def lidar_data_sort(self,position,angle,pointlist,width,length):
        coords = pointlist
        center = tuple(position)
        result=sorted(coords, key=lambda coord: (95-degrees(angle)-degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return result

    def lidar_sensor(self,position,angle,maxdistance,width,length,num_of_points=12):
        angle=radians(angle)
        lineimage=np.zeros([600,600])
        startpoint=(position[0]+maxdistance*math.cos(angle),position[1]-maxdistance*math.sin(angle))
        validindex=[]
        for i in range(num_of_points):
            newpoint=self.rotate((int(position[0]),int(position[1])),startpoint,radians(i*(-360/num_of_points)))
            for j in range(5):
                validindex+=[[newpoint[1],newpoint[0]]]
            lineimage=cv2.line(lineimage,(int(position[0]),int(position[1])),(int(newpoint[0]),int(newpoint[1])),255,2)
        validindex=np.array(validindex)
        contactpoint=bytescale(np.multiply(edge,lineimage))
        validindex=np.vstack([validindex,np.argwhere(contactpoint)])
        clustering = DBSCAN(eps=10, min_samples=2).fit(validindex)
        cluster_dict={}
        count=0
        for element in clustering.labels_:
            if element not in cluster_dict:
                cluster_dict[element]=[validindex[count]]
            else:
                cluster_dict[element]+=[validindex[count]]
            count+=1
        result=[]
        for element in cluster_dict:
            array=np.array(cluster_dict[element])
            result+=[[np.average(array[:,1]),np.average(array[:,0])]]
        C=position
        remove_list=[]
        for A in result:
            for B in result:
                A=np.array(A)
                B=np.array(B)
                if np.array_equal(A,B)==False :
                    maxlength=max(np.linalg.norm(A-B),np.linalg.norm(B-C),np.linalg.norm(C-A))
                    minheight=abs(A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1]))//maxlength
                    if minheight<5:
                        if abs(maxlength-np.linalg.norm(A-B))>1:
                            if np.linalg.norm(B-C)>=np.linalg.norm(A-C):
                                remove_list+=[list(B)]
                            else:
                                remove_list+=[list(A)]
        for element in remove_list:
            try:
                result.remove(element)
            except:
                pass
        while len(result)<num_of_points:
            result+=[position]
        if len(result)>num_of_points:
            result=result[:num_of_points]
        result=self.lidar_data_sort(position,angle,result,width,length)

        return result


    def check_collision(self,position,angle,width,length):
        x1,x2,x3,x4,y1,y2,y3,y4=self.find_carpoints(position,angle,width,length)
        x1=int(x1);x2=int(x2);x3=int(x3);x4=int(x4)
        y1=int(y1);y2=int(y2);y3=int(y3);y4=int(y4)
        cloneimage=np.array(image)
        #draw car rectangle
        cloneimage = cv2.line(cloneimage, (x1, y1), (x2, y2), (255, 255, 255))
        cloneimage = cv2.line(cloneimage, (x1, y1), (x3, y3), (255, 255, 255))
        cloneimage = cv2.line(cloneimage, (x4, y4), (x2, y2), (255, 255, 255))
        cloneimage = cv2.line(cloneimage, (x4, y4), (x3, y3), (255, 255, 255))
        #check sameness
        if np.array_equal(image,cloneimage):
            return False
        else:
            return True
    def find_carpoints(self,position,angle,width,length):
        #convert to radian
        angle=radians(angle)
        #lefttop
        x1=position[0]-width/2*math.cos(angle)-length/2*math.sin(angle)
        y1=position[1]+width/2*math.sin(angle)-length/2*math.cos(angle)
        #righttop
        x2=position[0]+width/2*math.cos(angle)-length/2*math.sin(angle)
        y2=position[1]-width/2*math.sin(angle)-length/2*math.cos(angle)
        #leftbottom
        x3=position[0]-width/2*math.cos(angle)+length/2*math.sin(angle)
        y3=position[1]+width/2*math.sin(angle)+length/2*math.cos(angle)
        #rightbottom
        x4=position[0]+width/2*math.cos(angle)+length/2*math.sin(angle)
        y4=position[1]-width/2*math.sin(angle)+length/2*math.cos(angle)

        return x1,x2,x3,x4,y1,y2,y3,y4

class Car:
    def __init__(self, x, y, angle=0.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.steering = 0.0
        self.rotation = 0.0


        self.vehicle="car"

        #car spec
        if self.vehicle=="car":
            self.scale=60/8.202
            self.carwidth=7.950*self.scale
            self.carlength=27.054*self.scale
            self.curb_to_curb_r=24.000*self.scale
            self.frontwheel_ratio=6.054/(self.carlength/self.scale)
            self.rearwheel_ratio=(6.054+13.083)/(self.carlength/self.scale)
        #spmt spec
        if self.vehicle=="spmt":
            self.carwidth=30
            self.carlength=150
        #trailer spec
        if self.vehicle=="trailer":
            self.carwidth=30
            self.carlength=30
            self.barwidth=10
            self.barlength=20
            self.rearwidth=30
            self.rearlength=60
            self.barposition = Vector2(x, y)
            self.barvelocity = Vector2(0.0, 0.0)
            self.rearposition = Vector2(x, y)
            self.rearvelocity = Vector2(0.0, 0.0)

    def update(self, dt,width,length,type="car"):
        if type=="car":
            """
            if self.steering:
                l_dot=length*(self.rearwheel_ratio-self.frontwheel_ratio)
                #24.5 , 12.1
                #angle : 36.91, 40.74
                #c-c radiaus : 40.026, 24.278
                #turning_radius = abs(tan(radians(self.steering)))/tan(radians(self.steering))*(math.sqrt(self.curb_to_curb_r*self.curb_to_curb_r-l_dot*l_dot)-width)
                turning_radius = abs(tan(radians(self.steering)))/tan(radians(self.steering))*24.9490*self.scale
                print(turning_radius/self.scale)
                angular_velocity = self.velocity.y / turning_radius
            else:
                angular_velocity = 0
            turning_center_ratio = (self.rearwheel_ratio+self.frontwheel_ratio)/2
            turning_length = math.sqrt((width/2)*(width/2)+length*(self.rearwheel_ratio-turning_center_ratio)*length*(self.rearwheel_ratio-turning_center_ratio))
            turning_vel=angular_velocity*turning_length
            theta=90-degrees(math.atan(length*(self.rearwheel_ratio-turning_center_ratio)/(width/2)))
            if self.steering>=0:
                turning_velocity=Vector2(-turning_vel*math.cos(radians(theta)),turning_vel*math.sin(radians(theta)))
            else:
                turning_velocity=Vector2(-turning_vel*math.cos(radians(theta)),-turning_vel*math.sin(radians(theta)))
            newposition=self.position + self.velocity.rotate(-self.angle) * dt + turning_velocity.rotate(-self.angle) * dt
            newangle=self.angle + degrees(angular_velocity) * dt
            """
            if self.steering:
                l_dot=length*(self.rearwheel_ratio-self.frontwheel_ratio)
                l_center_rear=length*(self.rearwheel_ratio-0.5)
                #24.5 , 12.1
                #angle : 36.91, 40.74
                #c-c radiaus : 40.026, 24.278
                #turning_radius = abs(tan(radians(self.steering)))/tan(radians(self.steering))*(math.sqrt(self.curb_to_curb_r*self.curb_to_curb_r-l_dot*l_dot)-width/2)

                """
                turning_radius=abs(tan(radians(self.steering)))/tan(radians(self.steering))*(self.curb_to_curb_r-self.carwidth/2)
                angular_velocity = self.velocity.y / turning_radius
                turning_center_ratio = (self.rearwheel_ratio+self.frontwheel_ratio)/2
                turning_length = math.sqrt((width/2)*(width/2)+length*(self.rearwheel_ratio-turning_center_ratio)*length*(self.rearwheel_ratio-turning_center_ratio))
                turning_vel=angular_velocity*turning_length
                theta=90-degrees(math.atan(length*(self.rearwheel_ratio-turning_center_ratio)/(width/2)))
                if self.steering>=0:
                    turning_velocity=Vector2(-turning_vel*math.cos(radians(theta)),turning_vel*math.sin(radians(theta)))
                else:
                    turning_velocity=Vector2(-turning_vel*math.cos(radians(theta)),-turning_vel*math.sin(radians(theta)))
                newposition=self.position + turning_velocity.rotate(-self.angle) * dt
                newangle=self.angle + degrees(angular_velocity) * dt
                """
                l_x = abs(tan(radians(self.steering)))/tan(radians(self.steering))*(math.sqrt(self.curb_to_curb_r*self.curb_to_curb_r-l_dot*l_dot)-width/2)
                turning_radius=abs(tan(radians(self.steering)))/tan(radians(self.steering))*(self.curb_to_curb_r-self.carwidth/2*(1/math.cos(radians(self.steering))))
                l_y = l_center_rear
                l_x_dot = l_x*math.cos(radians(-self.angle))-l_y*math.sin(radians(-self.angle))
                l_y_dot = l_x*math.sin(radians(-self.angle))+l_y*math.cos(radians(-self.angle))
                print(l_x,l_y,l_x_dot,l_y_dot)
                angular_velocity = self.velocity.y / turning_radius
                newposition=self.position + (l_x_dot*math.cos(angular_velocity*dt)-l_y_dot*math.sin(angular_velocity*dt),l_x_dot*math.sin(angular_velocity*dt)+l_y_dot*math.cos(angular_velocity*dt))-(l_x_dot,l_y_dot)
                newangle= self.angle + degrees(angular_velocity) * dt

            else:
                angular_velocity = 0
                newposition=self.position + self.velocity.rotate(-self.angle) * dt
                newangle=self.angle + degrees(angular_velocity) * dt
        elif type=="spmt":
            newposition=self.position + self.velocity.rotate(-self.angle) * dt
            newangle=self.angle+self.steering
        elif type=="trailer":
            pass

        check_collision=utility().check_collision(newposition,newangle,self.carwidth,self.carlength) or (newposition[0]<0 or newposition[0]>600 or newposition[1]<0 or newposition[1]>600)

        if check_collision:
            self.position = self.position
            self.angle =self.angle
            return 0
        else:
            self.position =newposition
            self.angle =newangle
            return 1



class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Swept Path Analysis")
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 10000
        self.exit = False

    def run(self):
        red = (255, 0, 0)
        green = (0,255,0)
        blue = (0,0,255)
        gray = (100,100,100)
        car = Car(x=391+30,y=372+101,angle=0)
        if car.vehicle=="car" or car.vehicle=="spmt":
            car_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
            car_image.fill(red)
            stack_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
            stack_image.fill(gray)
        elif car.vehicle=="trailer":
            car_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
            car_image.fill(red)
            bar_image = pygame.Surface((car.barwidth,car.barlength),pygame.SRCALPHA)
            bar_image.fill(green)
            rear_image=pygame.Surface((car.rearwidth,car.rearlength),pygame.SRCALPHA)
            rear_image.fill(blue)
            stack_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
            stack_image.fill(gray)


        road_image=pygame.image.load(imagefile)
        ppu = 1
        car_velocity=-100
        car_steering=36.91
        nextvalid=1
        stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position]]
        step_count=0
        while not self.exit:
            timemarker=time.time()
            #dt = self.clock.get_time() / 1000
            dt = 0.01
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_c]:
                pygame.image.save(self.screen, "screenshot.png")

            if pressed[pygame.K_UP]:
                car.velocity.y = car_velocity
            elif pressed[pygame.K_DOWN]:
                car.velocity.y = -car_velocity
            else :
                car.velocity.y = 0

            if pressed[pygame.K_RIGHT]:
                if car.vehicle=="car":
                    car.steering = car_steering
                elif car.vehicle=="spmt":
                    car.velocity.x=-car_velocity
            elif pressed[pygame.K_LEFT]:
                if car.vehicle=="car":
                    car.steering = -car_steering
                elif car.vehicle=="spmt":
                    car.velocity.x=car_velocity
            else :
                if car.vehicle=="car":
                    car.steering = 0
                elif car.vehicle=="spmt":
                    car.velocity.x=0

            if car.vehicle=="spmt":
                if pressed[pygame.K_q]:
                    car.steering=1
                elif pressed[pygame.K_e]:
                    car.steering=-1
                else:
                    car.steering=0
            # Logic
            result=[[car.position[1],car.position[0]]]

            # Distance Sensor
            num_of_points=16
            #if car.vehicle=="car" or car.vehicle=="spmt":
            #    result=utility().lidar_sensor(car.position,car.angle,1000,car.carwidth,car.carlength,num_of_points=num_of_points)
            result=[]
            if nextvalid==1:
                if (car.vehicle=="car" or car.vehicle=="spmt") and step_count!=1:
                    nextvalid=car.update(dt,car.carwidth,car.carlength,car.vehicle)
                elif (car.vehicle=="car" or car.vehicle=="spmt") and step_count==1:
                    car.position+=[0,-car.carlength]
                # Drawing
                self.screen.fill((0, 0, 0))
                rotated = pygame.transform.rotate(car_image, car.angle)
                rect = rotated.get_rect()
                self.screen.blit(road_image,(0,0))
                if [pygame.transform.rotate(stack_image,car.angle),car.position] not in stack_list:
                    stack_list+=[[pygame.transform.rotate(stack_image,car.angle),car.position]]
                for element in stack_list:
                    self.screen.blit(element[0], element[1] * ppu - (element[0].get_rect().width / 2, element[0].get_rect().height / 2))
                self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
                pygame.display.flip()
                step_count+=1
            else:
                # Initializing
                step_count=0
                nextvalid=1
                car = Car(x=391+30,y=372+101,angle=0)
                stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position]]
                # Drawing
                self.screen.fill((0, 0, 0))
                rotated = pygame.transform.rotate(car_image, car.angle)
                rect = rotated.get_rect()
                self.screen.blit(road_image,(0,0))
                self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
                pygame.display.flip()

            #print(time.time()-timemarker,len(result))
            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
