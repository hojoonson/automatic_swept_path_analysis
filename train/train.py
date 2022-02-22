from genericpath import exists
from utility import utility, Car
from collections import deque
from dqn import DeepQNetwork
import tensorflow as tf
import logging
from typing import List
import random
import os
import pygame
import cv2
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
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
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

    def step(self,action:int, car:Car) -> float:
        reward=0
        if car.vehicle=='car':
            car_steering=car.car_steering
            if car.rearvalid == 0:
                if action == 0:
                    car.velocity.y = car.car_velocity
                    car.steering = 0
                    reward = 0.1
                elif action == 1:
                    car.velocity.y = car.car_velocity
                    car.steering = car_steering
                    reward = 0.1
                elif action == 2:
                    car.velocity.y = car.car_velocity
                    car.steering = -car_steering
                    reward = 0.1
                elif action == 3:
                    reward = -0.2
                elif action == 4:
                    reward = -0.2
                elif action == 5:
                    reward = -0.2
            if car.rearvalid == 1:
                if action == 0:
                    car.velocity.y = car.car_velocity
                    car.steering = 0
                    reward = -0.05
                elif action == 1:
                    car.velocity.y = car.car_velocity
                    car.steering = car_steering
                    reward = -0.05
                elif action == 2:
                    car.velocity.y = car.car_velocity
                    car.steering = -car_steering
                    reward = -0.05
                elif action == 3:
                    car.velocity.y = -car.car_velocity
                    car.steering = 0
                    reward = -0.1
                elif action == 4:
                    car.velocity.y = -car.car_velocity
                    car.steering = car_steering
                    reward = 0.1
                elif action == 5:
                    car.velocity.y = -car.car_velocity
                    car.steering = -car_steering
                    reward = 0.1

        elif car.vehicle=='spmt':
            if car.rearvalid==0:
                if action==0:
                    car.velocity.x=0
                    car.velocity.y= car.car_velocity
                    car.steering = 0
                    reward=0.1
                elif action==1:
                    car.velocity.x=0
                    car.velocity.y=car.car_velocity
                    car.steering = 1
                    reward=0.1
                elif action==2:
                    car.velocity.x=0
                    car.velocity.y=car.car_velocity
                    car.steering = -1
                    reward=0.1
                elif action==3:
                    car.velocity.x=-car.car_velocity
                    car.velocity.y= car.car_velocity
                    car.steering=0
                    reward=0.1
                elif action==4:
                    car.velocity.x= car.car_velocity
                    car.velocity.y= car.car_velocity
                    car.steering=0
                    reward=0.1
                elif action==5:
                    reward=-0.2
                elif action==6:
                    reward=-0.2
                elif action==7:
                    reward=-0.2
                elif action==8:
                    reward=-0.2

            elif car.rearvalid==1:      
                if action==0:
                    car.velocity.x=0
                    car.velocity.y= car.car_velocity
                    car.steering = 0
                    reward=0.1
                elif action==1:
                    car.velocity.x=0
                    car.velocity.y=car.car_velocity
                    car.steering = 1
                    reward=0.1
                elif action==2:
                    car.velocity.x=0
                    car.velocity.y=car.car_velocity
                    car.steering = -1
                    reward=0.1
                elif action==3:
                    car.velocity.x=-car.car_velocity
                    car.velocity.y= car.car_velocity
                    car.steering=0
                    reward=0.1
                elif action==4:
                    car.velocity.x= car.car_velocity
                    car.velocity.y= car.car_velocity
                    car.steering=0
                    reward=0.1
                elif action==5:
                    car.velocity.x=0
                    car.velocity.y=-car.car_velocity
                    car.steering = 1
                    reward=0.1
                elif action==6:
                    car.velocity.x=0
                    car.velocity.y=-car.car_velocity
                    car.steering = -1
                    reward=0.1
                elif action==7:
                    car.velocity.x=-car.car_velocity
                    car.velocity.y=-car.car_velocity
                    car.steering=0
                    reward=0.1
                elif action==8:
                    car.velocity.x= car.car_velocity
                    car.velocity.y=-car.car_velocity
                    car.steering=0
                    reward=0.1
        return reward

    def action_sample(self,vehicle,outputsize):
        return random.randint(0,outputsize-1)
        

class Game:
    def __init__(self):
        self.vehicle_name = 'Mack_Trucks_TerraPro'
        self.vehicle_type='car'
        label_path=f'./data/train/trainlabels/{self.vehicle_name}_trainlabels.txt'
        f = open(label_path,'r')
        pathdata = f.readlines()
        path_list=[]
        for element in pathdata:
            if int(element.split(' ')[1])==1:
                path_list+=[element.split(' ')[0]]
        f.close()
        self.roadimage_path=path_list
        label_path=f'./data/test/testlabels/{self.vehicle_name}_testlabels.txt'

        f = open(label_path,'r')
        pathdata = f.readlines()
        path_list=[]
        for element in pathdata:
            if int(element.split(' ')[1])==1:
                path_list+=[element.split(' ')[0]]
        f.close()
        self.roadimage_path+=path_list
        print(self.roadimage_path)
        self.util=utility()
        self.random_candidate=3
        self.map_updatecount=3
        self.util.imagefile=self.roadimage_path[0]                    
        pygame.init()
        pygame.display.set_caption('Swept Path Analysis')
        self.startx=296.5
        self.starty=600
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
        else:
            self.outputsize=9
    def run(self):
        car = Car(x=self.startx,y=self.starty,angle=self.startangle,vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
        red = (255, 0, 0)
        gray = (100,100,100)
        car_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
        car_image.fill(red)
        stack_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
        stack_image.fill(gray)
        ppu = 1   
        if car.vehicle=='car':
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        elif car.vehicle=='spmt':
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        logger.info('FLAGS configure.')
        #logger.info(FLAGS.__flags)
        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
        os.makedirs('train_result', exist_ok=True)
        lossresult_path = f'./train_result/{self.vehicle_name}_output:{self.outputsize}_f{FLAGS.frame_size}_transport_{self.vehicle_type}_model_{FLAGS.model_name}_{FLAGS.checkpoint_path}_global_step'
        os.makedirs(lossresult_path, exist_ok=True)
        lossresult=open('./'+lossresult_path+'/loss.txt','w+')
        rewardresult_path = f'./train_result/{self.vehicle_name}_output:{self.outputsize}_f{FLAGS.frame_size}_transport_{self.vehicle_type}_model_{FLAGS.model_name}_{FLAGS.checkpoint_path}_global_step'
        os.makedirs(rewardresult_path, exist_ok=True)
        rewardresult=open('./'+rewardresult_path+'/reward.txt','w+')
        with tf.compat.v1.Session() as sess:
            mainDQN = DeepQNetwork(sess, FLAGS.model_name, train.input_size, train.output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name='main')
            targetDQN = DeepQNetwork(sess,FLAGS.model_name, train.input_size, train.output_size, frame_size=FLAGS.frame_size, name='target')

            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            #saver.restore(sess, tf.train.latest_checkpoint('Swept_Path_Analysis_byimage_output_10_f4_transport_spmt_model_Hojoon_Custom_CNN_forimage_v2_checkpoint'))

            # initial copy q_net -> target_net
            copy_ops = train.get_copy_var_ops(dest_scope_name='target',
                                        src_scope_name='main')
            sess.run(copy_ops)
            
            global_step = 1
            stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position,car.angle]]
            for episode in range(FLAGS.max_episode_count):
                #set road image!
                self.util.imagefile=self.roadimage_path[int(episode/self.map_updatecount)%len(self.roadimage_path)]
                print(self.util.imagefile)
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
                    car = Car(x=random_before_state[0][1][0], y=random_before_state[0][1][1], angle=random_before_state[0][2], vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
                else:
                    car = Car(x=self.startx, y=self.starty, angle=self.startangle, vehicle_type=self.vehicle_type, vehicle_name=self.vehicle_name)
                nextvalid=1
                stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position,car.angle]]
                self.done = False

                # Current State by image
                state=self.util.get_instant_image(car.position,car.angle,car.carwidth,car.carlength,self.scope_image_size,self.scope_image_resize)
                state=state.flatten()
                e_reward = 0
                model_loss = 0
                
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
                        action = train.action_sample(car.vehicle,self.outputsize)
                        #print('random action',action)
                    else:
                        # Get new state and reward from environment
                        action = np.argmax(mainDQN.predict(state))
                        #print('Q-func action',action)
                    

                    reward = train.step(action,car)
                    nextvalid,rear_count=car.update(dt,self.util.image,type=self.vehicle_type,rear_count=rear_count)
                    event=pygame.event.get()
            
                    if len(event)!=0:
                        if event[0].type==pygame.KEYDOWN:
                            if event[0].key==pygame.K_q:
                                nextvalid=3
                    same_check_list.append(np.array(car.position))
                    if nextvalid==0:
                        #print('Collision!!!')
                        reward=-4.0
                    if nextvalid==2:
                        print('Finish the Analysis!!!')
                        reward=0.4
                    if nextvalid==3:
                        print('force quit to next episode')
                    # Current State by Lidar Sensor
                    next_state=self.util.get_instant_image(car.position,car.angle,car.carwidth,car.carlength,self.scope_image_size,self.scope_image_resize)
                    next_state=next_state.flatten()
                    if FLAGS.frame_size > 1:
                        state_with_frame.append(next_state)
                        next_state = np.array(state_with_frame)
                        next_state = np.reshape(next_state, (1, train.RAM_FIXED_LENGTH, FLAGS.frame_size))
                    # Save the experience to our buffer
                    replay_buffer.append((state, action, reward, next_state, self.done))

                    if len(replay_buffer) > FLAGS.batch_size:
                        minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                        loss, _ = train.replay_train(mainDQN, targetDQN, minibatch)
                        model_loss = loss
                        if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                            print(' - step_count : '+str(step_count)+', reward: '+str(e_reward)+' ,loss: '+str(loss))
                        if global_step % 100 == 0:
                            lossresult.write('global_step:'+str(global_step)+ ' loss: '+str(loss)+'\n')
                    if step_count % FLAGS.target_update_count == 0:
                        sess.run(copy_ops)

                    state = next_state
                    e_reward += reward
                    step_count += 1

                    if len(same_check_list)==50 and len(np.unique(same_check_list))==2:
                        rewardresult.write('episode:'+str(episode)+ ' reward: '+str(e_reward)+'\n')
                        self.done=True
                    if (nextvalid!=1 and nextvalid!=0) or (step_count!=0 and step_count%1000==0):
                        rewardresult.write('episode:'+str(episode)+ ' reward: '+str(e_reward)+'\n')
                        self.done=True

                    # save model checkpoint
                    if global_step % FLAGS.save_step_count == 0:
                        checkpoint_path = f'./train_result/{self.vehicle_name}_output:{self.outputsize}_f{FLAGS.frame_size}_transport_{self.vehicle_type}_model_{FLAGS.model_name}_{FLAGS.checkpoint_path}_global_step'
                        os.makedirs(checkpoint_path, exist_ok=True)
                        saver.save(sess, checkpoint_path, global_step=global_step)
                        logger.info('save model for global_step: '+str(global_step))

                    # Drawing
                    self.screen.fill((0, 0, 0))
                    rotated = pygame.transform.rotate(car_image, car.angle)
                    rect = rotated.get_rect()
                    self.screen.blit(road_image,(0,0))
                    fontObj = pygame.font.Font('./font/times-new-roman.ttf', 30)
                    textSurfaceObj = fontObj.render('Episode '+str(episode), True, (255,255,255), (0,0,0))
                    textRectObj = textSurfaceObj.get_rect()
                    textRectObj.center = (100,30)
                    self.screen.blit(textSurfaceObj, textRectObj) 
                    count=1

                    self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2)) 
                    pygame.display.flip()
                    #print(time.time()-timemarker,len(result))
                    #self.clock.tick(self.ticks)
                    global_step+=1
            lossresult.close()
            rewardresult.close()
            pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
