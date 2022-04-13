from utility import utility, Vehicle
from collections import deque
from dqn import DeepQNetwork, update_action_and_get_reward
import tensorflow as tf
import logging
from typing import List
import random
import os
import pygame
import cv2
import datetime
import csv
import numpy as np
from pygame.locals import HWSURFACE, DOUBLEBUF

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

flags = tf.compat.v1.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 3000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 15000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 32, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 4, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'Custom_CNN_forimage_v2', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.000001, 'Learning rate. ')
flags.DEFINE_boolean('step_verbose', True, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 50, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
FLAGS = flags.FLAGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Train:
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.RAM_FIXED_LENGTH=input_size[0]

    def replay_train(self,mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
        states = np.vstack([x[0] for x in train_batch])
        actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
        rewards = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])
        next_states = np.vstack([x[3] for x in train_batch])
        done = np.array([x[4] for x in train_batch[:FLAGS.batch_size]])
        predict_result = targetDQN.predict(next_states)
        Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1) * (1 - done)

        X = states
        y = mainDQN.predict(states)
        y[np.arange(len(X)), actions] = Q_target
        # Train our network using target and predicted Q values on each episode
        return mainDQN.update(X, y)

    def get_copy_var_ops(self,*, dest_scope_name: str, src_scope_name: str) -> List[tf.compat.v1.Operation]:
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def action_sample(self,vehicle,outputsize):
        return random.randint(0,outputsize-1)
        

class Simulation:
    def __init__(self):
        self.vehicle_name = 'v1'
        self.vehicle_type='spmt'
        label_path='./generation_result/total_test_data_0411/manual_labelling_result/v1_total.txt'
        # label_path=os.path.join('data','train','trainlabels',f'{self.vehicle_name}_trainlabels.txt')
        path_list=[]
        with open(label_path,'r') as f:
            pathdata = f.readlines()
            for element in pathdata:
                splited = element.split(' ')
                path_list.append({
                    'image_path': splited[0],
                    'startx': float(splited[1]),
                    'starty': float(splited[2]),
                    'retry': 0,
                    'passcnt':0,
                    'label': '0',
                    'gt': splited[3],
                    'correct': '0' == splited[3]
                })
        random.shuffle(path_list)
        self.roadimage_path=path_list
        print(self.roadimage_path)
        self.util=utility()
        self.random_candidate=3
        self.map_updatecount=2
        self.multiple = 5
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
        if self.vehicle_type == 'car':
            self.outputsize=6
        elif self.vehicle_type == 'spmt':
            self.outputsize=9
    def run(self):
        vehicle = Vehicle(x=self.startx,y=self.starty,angle=self.startangle,vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
        red = (255, 0, 0)
        gray = (100,100,100)
        car_image = pygame.Surface((vehicle.carwidth,vehicle.carlength),pygame.SRCALPHA)
        car_image.fill(red)
        stack_image = pygame.Surface((vehicle.carwidth,vehicle.carlength),pygame.SRCALPHA)
        stack_image.fill(gray)
        ppu = 1   
        if vehicle.vehicle=='car':
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        elif vehicle.vehicle=='spmt':
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        logger.info('FLAGS configure.')
        #logger.info(FLAGS.__flags)
        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
        
        # Initialize result paths
        os.makedirs('automatic_labelling_result', exist_ok=True)
        save_time = str(datetime.datetime.now())
        result_path = os.path.join('automatic_labelling_result',f'{self.vehicle_name}_output:{self.outputsize}_f{FLAGS.frame_size}_transport_{self.vehicle_type}_model_{FLAGS.model_name}_checkpoint')
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, save_time)
        os.makedirs(result_path, exist_ok=True)
        checkpoint_path = os.path.join(result_path, '_global_step')
        result_path = os.path.join(result_path, 'result')
        os.makedirs(result_path, exist_ok=True)

        with open(os.path.join(result_path,'reward_and_loss.csv'), 'w', newline='') as reward_and_loss:
            resultwriter = csv.writer(reward_and_loss)
            resultwriter.writerow(['Episode', 'Reward', 'Loss'])

        with tf.compat.v1.Session() as sess:
            mainDQN = DeepQNetwork(sess, FLAGS.model_name, train.input_size, train.output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name='main')
            targetDQN = DeepQNetwork(sess,FLAGS.model_name, train.input_size, train.output_size, frame_size=FLAGS.frame_size, name='target')

            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            #saver.restore(sess, tf.train.latest_checkpoint('Swept_Path_Analysis_byimage_output_10_f4_transport_spmt_model_Hojoon_Custom_CNN_forimage_v2_checkpoint'))

            # initial copy q_net -> target_net
            copy_ops = train.get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
            sess.run(copy_ops)
            
            global_step = 1
            stack_list=[[pygame.transform.rotate(stack_image,vehicle.angle),vehicle.position,vehicle.angle]]
            for episode in range(self.map_updatecount*len(self.roadimage_path)*self.multiple):
                #set road image!
                index = int(episode/self.map_updatecount)%len(self.roadimage_path)
                if self.roadimage_path[index]['passcnt'] > self.map_updatecount or\
                    self.roadimage_path[index]['retry'] > self.map_updatecount * self.multiple:
                    continue
                self.util.imagefile=self.roadimage_path[index]['image_path']
                self.startx=self.roadimage_path[index]['startx']
                self.starty=self.roadimage_path[index]['starty']
                logger.info(f'{self.util.imagefile}, {self.startx}, {self.starty}')
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
                    vehicle = Vehicle(x=random_before_state[0][1][0], y=random_before_state[0][1][1], angle=random_before_state[0][2], vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
                else:
                    vehicle = Vehicle(x=self.startx, y=self.starty, angle=self.startangle, vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
                nextvalid=1
                stack_list=[[pygame.transform.rotate(stack_image,vehicle.angle),vehicle.position,vehicle.angle]]
                self.done = False

                # Current State by image
                state=self.util.get_instant_image(vehicle.position,vehicle.angle,vehicle.carwidth,vehicle.carlength,self.scope_image_size,self.scope_image_resize)
                state=state.flatten()
                e_reward = 0
                e_loss = 0
                e_step = 0
                if FLAGS.frame_size > 1:
                    state_with_frame = deque(maxlen=FLAGS.frame_size)
                    for _ in range(FLAGS.frame_size):
                        state_with_frame.append(state)
                    state = np.array(state_with_frame)
                    state = np.reshape(state, (1,train.RAM_FIXED_LENGTH, FLAGS.frame_size))
                
                same_check_list=deque(maxlen=50)
                rear_count=0
                while not self.done:
                    #dt = self.clock.get_time() / 1000
                    if self.vehicle_type=='car':
                        dt = 0.03
                    elif self.vehicle_type=='spmt':
                        dt = 0.04
                    if np.random.rand() < e:
                        # random action
                        action = train.action_sample(vehicle.vehicle,self.outputsize)
                    else:
                        # Get new state and reward from environment
                        action = np.argmax(mainDQN.predict(state))
                    

                    reward = update_action_and_get_reward(action, vehicle)
                    nextvalid,rear_count=vehicle.update(dt,self.util.image,type=self.vehicle_type,rear_count=rear_count)
                    event=pygame.event.get()
            
                    if len(event)!=0:
                        if event[0].type==pygame.KEYDOWN:
                            if event[0].key==pygame.K_q:
                                nextvalid=3
                    same_check_list.append(np.array(vehicle.position))
                    if nextvalid==0:
                        #logger.info('Collision!!!')
                        reward = min(-4.0, reward - 4.0)
                    if nextvalid==2:
                        if vehicle.velocity.y<=0:
                            logger.info('Finish the Analysis!!!')
                            reward = max(4.0, reward + 4.0)
                        else:
                            nextvalid = 0
                            reward = min(-4.0, reward - 4.0)
                    if nextvalid==3:
                        logger.info('force quit to next episode')

                    # Next State
                    next_state=self.util.get_instant_image(vehicle.position,vehicle.angle,vehicle.carwidth,vehicle.carlength,self.scope_image_size,self.scope_image_resize)
                    next_state=next_state.flatten()
                    if FLAGS.frame_size > 1:
                        state_with_frame.append(next_state)
                        next_state = np.array(state_with_frame)
                        next_state = np.reshape(next_state, (1, train.RAM_FIXED_LENGTH, FLAGS.frame_size))
                    # Save the experience to buffer
                    replay_buffer.append((state, action, reward, next_state, self.done))

                    if len(replay_buffer) > FLAGS.batch_size:
                        minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                        loss, _ = train.replay_train(mainDQN, targetDQN, minibatch)
                        e_loss += loss
                        e_step += 1
                        if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                            logger.info(f'step_count : {step_count}, reward: {e_reward}, loss: {loss}')
                    if step_count % FLAGS.target_update_count == 0:
                        sess.run(copy_ops)

                    state = next_state
                    e_reward += reward
                    step_count += 1

                    if len(same_check_list)==50 and len(np.unique(same_check_list))==2:
                        self.roadimage_path[index]['retry'] += 1
                        e_loss /= e_step
                        with open(os.path.join(result_path,'reward_and_loss.csv'), 'a', newline='') as reward_and_loss:
                            resultwriter = csv.writer(reward_and_loss)
                            resultwriter.writerow([episode, e_reward, e_loss])
                        with open(os.path.join(result_path,'label.csv'), 'w', newline='') as output_file:
                            dict_writer = csv.DictWriter(output_file, self.roadimage_path[index].keys())
                            dict_writer.writeheader()
                            dict_writer.writerows(self.roadimage_path)
                        self.done=True
                    if (nextvalid!=1 and nextvalid!=0) or (step_count!=0 and step_count%1000==0):
                        if nextvalid==2:
                            self.roadimage_path[index]['label'] = '1'
                            self.roadimage_path[index]['passcnt'] +=1
                            self.roadimage_path[index]['correct'] = '1' == self.roadimage_path[index]['gt']
                        else:
                            self.roadimage_path[index]['retry'] += 1
                        e_loss /= e_step
                        with open(os.path.join(result_path,'reward_and_loss.csv'), 'a', newline='') as reward_and_loss:
                            resultwriter = csv.writer(reward_and_loss)
                            resultwriter.writerow([episode, e_reward, e_loss])
                        with open(os.path.join(result_path,'label.csv'), 'w', newline='') as output_file:
                            dict_writer = csv.DictWriter(output_file, self.roadimage_path[index].keys())
                            dict_writer.writeheader()
                            dict_writer.writerows(self.roadimage_path)
                        self.done=True



                    # save model checkpoint
                    if global_step % FLAGS.save_step_count == 0:
                        saver.save(sess, checkpoint_path, global_step=global_step)
                        logger.info(f'save model for global_step: {global_step}')

                    # Drawing
                    self.screen.fill((0, 0, 0))
                    rotated = pygame.transform.rotate(car_image, vehicle.angle)
                    rect = rotated.get_rect()
                    self.screen.blit(road_image,(0,0))
                    fontObj = pygame.font.Font('./font/times-new-roman.ttf', 30)
                    textSurfaceObj = fontObj.render('Episode '+str(episode), True, (255,255,255), (0,0,0))
                    textRectObj = textSurfaceObj.get_rect()
                    textRectObj.center = (100,30)
                    self.screen.blit(textSurfaceObj, textRectObj) 

                    self.screen.blit(rotated, vehicle.position * ppu - (rect.width / 2, rect.height / 2)) 
                    pygame.display.flip()
                    #logger.info(time.time()-timemarker,len(result))
                    #self.clock.tick(self.ticks)
                    global_step+=1
            pygame.quit()


if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
