from utility import utility, Vehicle
from collections import deque
import logging
import random
import os
import time
import pygame
import cv2
import datetime
import numpy as np
from operator import itemgetter
from pygame.locals import HWSURFACE, DOUBLEBUF

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

max_episode_count = 10

class Game:
    def __init__(self):
        label_path=os.path.join('sample_data','sample_label.txt')
        self.vehicle_name='Scherule'
        self.vehicle_type='spmt'
        self.save_path=os.path.join(os.path.dirname(label_path),
                                    f'{self.vehicle_name}_{self.vehicle_type}_label_{str(datetime.datetime.now())}.txt')
        path_list=[]
        with open(label_path,'r') as f:
            pathdata = f.readlines()
            for element in pathdata:
                splited = element.split(' ')
                path_list.append({
                    'image_path': splited[0],
                    'startx': float(splited[1]),
                    'starty': float(splited[2])
                })
        self.roadimage_path=sorted(path_list, key=itemgetter('image_path'))
        self.util=utility()
        self.random_candidate=2
        self.map_updatecount=1
        self.util.imagefile=self.roadimage_path[0]['image_path']
        pygame.init()
        pygame.display.set_caption('Swept Path Analysis')
        self.startx=self.roadimage_path[0]['startx']
        self.starty=self.roadimage_path[0]['starty']
        self.startangle=0
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height), HWSURFACE | DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.ticks = 1000
        self.scope_image_size=300
        self.scope_image_resize=64
    
    def run(self):
        vehicle = Vehicle(x=self.startx,y=self.starty,angle=self.startangle,vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
        red = (255, 0, 0)
        gray = (100,100,100)
        car_image = pygame.Surface((vehicle.carwidth,vehicle.carlength),pygame.SRCALPHA)
        car_image.fill(red)
        stack_image = pygame.Surface((vehicle.carwidth,vehicle.carlength),pygame.SRCALPHA)
        stack_image.fill(gray)
        ppu = 1   

        logger.info('FLAGS configure.')
        # store the previous observations in replay memory
        global_step = 1
        stack_list=[[pygame.transform.rotate(stack_image,vehicle.angle),vehicle.position,vehicle.angle]]
        breakvalid=0
        presscount=0
        trainlabel_path = 'testlabels'
        if not os.path.exists(trainlabel_path):
            os.makedirs(trainlabel_path)

        truecount=0
        falsecount=0
        for episode in range(max_episode_count):
            logger.info('New episode start!!')
            if breakvalid==1:
                pygame.quit()
                break
            #set road image!
            index=int(episode/self.map_updatecount)%len(self.roadimage_path)
            self.util.imagefile=self.roadimage_path[index]['image_path']
            self.startx=self.roadimage_path[index]['startx']
            self.starty=self.roadimage_path[index]['starty']
            self.util.image=cv2.imread(self.util.imagefile)
            self.util.edge=cv2.Laplacian(self.util.image,cv2.CV_8U)
            self.util.edge=cv2.cvtColor(self.util.edge,cv2.COLOR_BGR2GRAY)
            road_image=pygame.image.load(self.util.imagefile)
            
            #define epcilon. it decay from 0.9 to 0.2
            e = 1. / ((episode / 20) + 1)
            step_count = 0
            #reset collision, finish valid
            self.collision_valid=0
            self.finish_valid=0
            # Initializing
            if len(stack_list)>=self.random_candidate and episode%self.map_updatecount!=0:
                randomstate=random.sample(stack_list,self.random_candidate)
                randomstate+=[[0,[self.startx,self.starty],self.startangle]]
                random_before_state=random.sample(randomstate,1)
                vehicle = Vehicle(x=random_before_state[0][1][0],y=random_before_state[0][1][1],angle=random_before_state[0][2],vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
            else:
                vehicle = Vehicle(x=self.startx,y=self.starty,angle=self.startangle,vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
            nextvalid=1
            stack_list=[[pygame.transform.rotate(stack_image,vehicle.angle),vehicle.position,vehicle.angle]]
            self.done = False

            
            same_check_list=deque(maxlen=50)
            rear_count=0
            while not self.done:
                #dt = self.clock.get_time() / 1000
                if self.vehicle_type=='car':
                    dt = 0.04
                elif self.vehicle_type=='spmt':
                    dt = 0.04
                # Event queue
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True
                # User input 
                action=-1
                pressed = pygame.key.get_pressed()
                vehicle.velocity.x=0
                vehicle.velocity.y=0
                vehicle.steering=0
                if pressed[pygame.K_c]:
                    pygame.image.save(self.screen, 'screenshot.png')
                if pressed[pygame.K_UP]:
                    action=0
                    if pressed[pygame.K_RIGHT]:
                        if vehicle.vehicle=='car':
                            action=1
                        elif vehicle.vehicle=='spmt':
                            action=1
                    elif pressed[pygame.K_LEFT]:
                        if vehicle.vehicle=='car':
                            action=2
                        elif vehicle.vehicle=='spmt':
                            action=2
                    elif pressed[pygame.K_q]:
                        if vehicle.vehicle=='spmt':
                            action=10
                    elif pressed[pygame.K_e]:
                        if vehicle.vehicle=='spmt':
                            action=11
                    else :
                        if vehicle.vehicle=='car':
                            vehicle.velocity.x=0
                            vehicle.velocity.y=0
                            vehicle.steering=0
                        elif vehicle.vehicle=='spmt':
                            vehicle.velocity.x=0
                            vehicle.velocity.y=0
                            vehicle.steering=0

                elif pressed[pygame.K_DOWN]:
                    action=3
                    if pressed[pygame.K_RIGHT]:
                        if vehicle.vehicle=='car':
                            action=4
                        elif vehicle.vehicle=='spmt':
                            action=4
                    elif pressed[pygame.K_LEFT]:
                        if vehicle.vehicle=='car':
                            action=5
                        elif vehicle.vehicle=='spmt':
                            action=5
                    elif pressed[pygame.K_q]:
                        if vehicle.vehicle=='spmt':
                            action=12
                    elif pressed[pygame.K_e]:
                        if vehicle.vehicle=='spmt':
                            action=13
                    else:
                        if vehicle.vehicle=='car':
                            vehicle.velocity.x=0
                            vehicle.velocity.y=0
                            vehicle.steering=0
                        elif vehicle.vehicle=='spmt':
                            vehicle.velocity.x=0
                            vehicle.velocity.y=0
                            vehicle.steering=0
                else:
                    if vehicle.vehicle=='spmt':
                        if pressed[pygame.K_q]:
                            action=6
                        elif pressed[pygame.K_e]:
                            action=7
                        elif pressed[pygame.K_RIGHT]:
                            action=8
                        elif pressed[pygame.K_LEFT]:
                            action=9
                        else:
                            vehicle.velocity.x=0
                            vehicle.velocity.y=0
                            vehicle.steering=0

                if pressed[pygame.K_t] and presscount<=1000:
                    if 0<presscount<=999:
                        presscount+=1
                        continue
                    elif presscount==1000:
                        presscount=0
                        continue
                    elif presscount==0:
                        presscount+=1
                        logger.info(self.util.imagefile, True)
                        logger.info(f'{os.path.splitext(self.util.imagefile)[0]}_1.png', True)
                        #trainlabel_element=open('./'+trainlabel_path+'/Sheurle_6axle_hojoon.txt','a+')
                        #trainlabel_element.write(self.util.imagefile+' 1 \n')
                        #trainlabel_element.write(self.util.imagefile.split('.png')[0]+'_1.png'+' 1 \n')
                        #trainlabel_element.close()
                        truecount+=1
                        logger.info(f'True : {truecount} False : {falsecount}')
                        time.sleep(0.2)
                        self.done=True

                elif pressed[pygame.K_f] and presscount<=1000:
                    if 0<presscount<=999:
                        presscount+=1
                        continue
                    elif presscount==1000:
                        presscount=0
                        continue
                    elif presscount==0:
                        presscount+=1
                        logger.info(self.util.imagefile, False)
                        logger.info(f'{os.path.splitext(self.util.imagefile)[0]}_1.png',False)
                        falsecount+=1
                        logger.info(f'True : {truecount} False : {falsecount}')
                        time.sleep(0.2)
                        self.done=True
                vehicle.step(action)
                nextvalid,rear_count=vehicle.update(dt,self.util.image,type=self.vehicle_type,rear_count=rear_count)

                if (nextvalid!=1 and nextvalid!=0):
                    self.done=True
                elif nextvalid==2:
                    logger.info(self.util.imagefile, True)
                    logger.info(f'{os.path.splitext(self.util.imagefile)[0]}_1.png', True)
                    #trainlabel_element=open('./'+trainlabel_path+'/Sheurle_6axle_hojoon.txt','a+')
                    #trainlabel_element.write(self.util.imagefile+' 1 \n')
                    #trainlabel_element.write(self.util.imagefile.split('.png')[0]+'_1.png'+' 1 \n')
                    #trainlabel_element.close()
                    truecount+=1
                    logger.info(f'True: {truecount} | False: {falsecount}')

                # Current State by Image
                next_state=self.util.get_instant_image(vehicle.position,vehicle.angle,vehicle.carwidth,vehicle.carlength,self.scope_image_size,self.scope_image_resize)
                next_state=next_state.flatten()
                step_count += 1
                # Drawing
                self.screen.fill((0, 0, 0))
                rotated = pygame.transform.rotate(car_image, vehicle.angle)
                rect = rotated.get_rect()
                self.screen.blit(road_image,(0,0))
                
                element = [pygame.transform.rotate(stack_image,vehicle.angle),vehicle.position,vehicle.angle]
                road_image.blit(element[0], element[1] * ppu - (element[0].get_rect().width / 2, element[0].get_rect().height / 2))
                
                '''writing episode'''
                fontObj = pygame.font.Font('./font/times-new-roman.ttf', 30)
                textSurfaceObj = fontObj.render('Episode '+str(episode), True, (255,255,255), (0,0,0))
                textRectObj = textSurfaceObj.get_rect()
                textRectObj.center = (100,30)
                self.screen.blit(textSurfaceObj, textRectObj) 

                self.screen.blit(rotated, vehicle.position * ppu - (rect.width / 2, rect.height / 2)) 
                pygame.display.flip()
                global_step+=1
if __name__ == '__main__':
    game = Game()
    game.run()
