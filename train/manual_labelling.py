from utility import *
from collections import deque
from dqn import DeepQNetwork
import tensorflow as tf
import logging
from typing import List
import random
import glob
import os
import tqdm
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 20000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 15000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 128, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 4, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'Hojoon_Custom_CNN_forimage_v2', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.000001, 'Learning rate. ')
flags.DEFINE_boolean('step_verbose', True, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 50, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
flags.DEFINE_integer('pretrain_iteration',100000,'pretrain iteration')
FLAGS = flags.FLAGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Train:
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.RAM_FIXED_LENGTH=input_size[0]
    def replay_train(self,mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
        """Trains `mainDQN` with target Q values given by `targetDQN`
        Args:
            mainDQN (DeepQNetwork``): Main DQN that will be trained
            targetDQN (DeepQNetwork): Target DQN that will predict Q_target
            train_batch (list): Minibatch of replay memory
                Each element is (s, a, r, s', done)
                [(state, action, reward, next_state, done), ...]
        Returns:
            float: After updating `mainDQN`, it returns a `loss`
        """
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

    def get_copy_var_ops(self,*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
        """Creates TF operations that copy weights from `src_scope` to `dest_scope`
        Args:
            dest_scope_name (str): Destination weights (copy to)
            src_scope_name (str): Source weight (copy from)
        Returns:
            List[tf.Operation]: Update operations are created and returned
        """
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder
    def step(self,action,car):
        reward=0
        #Mack Trucks TerraPro Low Entry 4x2 LEU612
        #car_steering=39.16
        #Pantechnicon_Removals_Van
        if car.vehicle=="car":
            car_steering=36.91

            if car.rearvalid==0:
                if action==0:
                    car.velocity.y= car.car_velocity
                    car.steering = 0
                elif action==1:
                    car.velocity.y = car.car_velocity
                    car.steering = car_steering
                elif action==2:
                    car.velocity.y= car.car_velocity
                    car.steering = -car_steering
                elif action==3:
                    car.velocity.y= -car.car_velocity
                    car.steering = 0
                elif action==4:
                    car.velocity.y = -car.car_velocity
                    car.steering = car_steering
                elif action==5:
                    car.velocity.y= -car.car_velocity
                    car.steering = -car_steering
            if car.rearvalid==1:
                if action==0:
                    car.velocity.y= car.car_velocity
                    car.steering = 0
                elif action==1:
                    car.velocity.y = car.car_velocity
                    car.steering = car_steering
                elif action==2:
                    car.velocity.y= car.car_velocity
                    car.steering = -car_steering
                elif action==3:
                    car.velocity.y= -car.car_velocity
                    car.steering = 0
                elif action==4:
                    car.velocity.y = -car.car_velocity
                    car.steering = car_steering
                elif action==5:
                    car.velocity.y= -car.car_velocity
                    car.steering = -car_steering
        elif car.vehicle=="spmt":
            if action==0:
                car.velocity.x=0
                car.velocity.y= car.car_velocity
                car.steering = 0
            elif action==1:
                car.velocity.x=-car.car_velocity
                car.velocity.y= car.car_velocity
                car.steering=0
            elif action==2:
                car.velocity.x= car.car_velocity
                car.velocity.y= car.car_velocity
                car.steering=0
            elif action==3:
                car.velocity.x=0
                car.velocity.y=-car.car_velocity
                car.steering=0
            elif action==4:
                car.velocity.x=-car.car_velocity
                car.velocity.y=-car.car_velocity
                car.steering=0
            elif action==5:
                car.velocity.x= car.car_velocity
                car.velocity.y=-car.car_velocity
                car.steering=0
            elif action==6:
                car.velocity.x=0
                car.velocity.y=0
                car.steering = 1
            elif action==7:
                car.velocity.x=0
                car.velocity.y=0
                car.steering = -1
            elif action==8:
                car.velocity.x=-car.car_velocity
                car.velocity.y= 0
                car.steering=0
            elif action==9:
                car.velocity.x= car.car_velocity
                car.velocity.y= 0
                car.steering=0

            elif action==10:
                car.velocity.x=0
                car.velocity.y=car.car_velocity
                car.steering = 1
            elif action==11:
                car.velocity.x=0
                car.velocity.y=car.car_velocity
                car.steering = -1
            elif action==12:
                car.velocity.x= 0
                car.velocity.y=-car.car_velocity
                car.steering= 1
            elif action==13:
                car.velocity.x= 0
                car.velocity.y= -car.car_velocity
                car.steering= -1
        return reward

    def action_sample(self,vehicle,outputsize):
        return random.randint(0,outputsize-1)
class Game:
    def __init__(self):
        self.roadimage_path=glob.glob("./manual_test_image/*.png")
        self.roadimage_path.sort()
        self.util=utility()
        self.random_candidate=2
        self.map_updatecount=1
        self.util.imagefile=self.roadimage_path[0]                    
        pygame.init()
        pygame.display.set_caption("Swept Path Analysis")
        self.startx=295.5
        self.starty=600
        self.startangle=0
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height), HWSURFACE | DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.ticks = 1000
        self.vehicle="spmt"
        self.scope_image_size=300
        self.scope_image_resize=64
        self.outputsize=6
        #spmt : 5
        #car : 3
    
    def run(self):
        car = Car(x=self.startx,y=self.starty,angle=self.startangle,vehicle=self.vehicle)
        red = (255, 0, 0)
        gray = (100,100,100)
        car_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
        car_image.fill(red)
        stack_image = pygame.Surface((car.carwidth,car.carlength),pygame.SRCALPHA)
        stack_image.fill(gray)
        ppu = 1   
        if car.vehicle=="car":
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        elif car.vehicle=="spmt":
            train=Train(input_size=(self.scope_image_resize*self.scope_image_resize,),output_size=self.outputsize)
        logger.info("FLAGS configure.")
        #logger.info(FLAGS.__flags)
        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
        global_step = 1
        stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position,car.angle]]
        breakvalid=0
        presscount=0
        trainlabel_path = "testlabels"
        if not os.path.exists(trainlabel_path):
            os.makedirs(trainlabel_path)

        truecount=0
        falsecount=0
        for episode in range(FLAGS.max_episode_count):
            print("New episode start!!")
            if breakvalid==1:
                pygame.quit()
                break
            #set road image!
            self.util.imagefile=self.roadimage_path[int(episode/self.map_updatecount)%len(self.roadimage_path)]
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
                car = Car(x=random_before_state[0][1][0],y=random_before_state[0][1][1],angle=random_before_state[0][2],vehicle=self.vehicle)
            else:
                car = Car(x=self.startx,y=self.starty,angle=self.startangle,vehicle=self.vehicle)
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
                if self.vehicle=="car":
                    dt = 0.04
                elif self.vehicle=="spmt":
                    dt = 0.04
                # Event queue
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True
                # User input 
                action=-1
                pressed = pygame.key.get_pressed()
                car.velocity.x=0
                car.velocity.y=0
                car.steering=0
                if pressed[pygame.K_c]:
                    pygame.image.save(self.screen, "screenshot.png")
                if pressed[pygame.K_UP]:
                    action=0
                    if pressed[pygame.K_RIGHT]:
                        if car.vehicle=="car":
                            action=1
                        elif car.vehicle=="spmt":
                            action=1
                    elif pressed[pygame.K_LEFT]:
                        if car.vehicle=="car":
                            action=2
                        elif car.vehicle=="spmt":
                            action=2
                    elif pressed[pygame.K_q]:
                        if car.vehicle=="spmt":
                            action=10
                    elif pressed[pygame.K_e]:
                        if car.vehicle=="spmt":
                            action=11
                    else :
                        if car.vehicle=="car":
                            car.velocity.x=0
                            car.velocity.y=0
                            car.steering=0
                        elif car.vehicle=="spmt":
                            car.velocity.x=0
                            car.velocity.y=0
                            car.steering=0

                elif pressed[pygame.K_DOWN]:
                    action=3
                    if pressed[pygame.K_RIGHT]:
                        if car.vehicle=="car":
                            action=4
                        elif car.vehicle=="spmt":
                            action=4
                    elif pressed[pygame.K_LEFT]:
                        if car.vehicle=="car":
                            action=5
                        elif car.vehicle=="spmt":
                            action=5
                    elif pressed[pygame.K_q]:
                        if car.vehicle=="spmt":
                            action=12
                    elif pressed[pygame.K_e]:
                        if car.vehicle=="spmt":
                            action=13
                    else :
                        if car.vehicle=="car":
                            car.velocity.x=0
                            car.velocity.y=0
                            car.steering=0
                        elif car.vehicle=="spmt":
                            car.velocity.x=0
                            car.velocity.y=0
                            car.steering=0
                else  :
                    if car.vehicle=="spmt":
                        if pressed[pygame.K_q]:
                            action=6
                        elif pressed[pygame.K_e]:
                            action=7
                        elif pressed[pygame.K_RIGHT]:
                            action=8
                        elif pressed[pygame.K_LEFT]:
                            action=9
                        else:
                            car.velocity.x=0
                            car.velocity.y=0
                            car.steering=0

                if pressed[pygame.K_t] and presscount<=1000:
                    if 0<presscount<=999:
                        presscount+=1
                        continue
                    elif presscount==1000:
                        presscount=0
                        continue
                    elif presscount==0:
                        presscount+=1
                        print(self.util.imagefile,"True")
                        print(self.util.imagefile.split(".png")[0]+"_1.png","True")
                        #trainlabel_element=open("./"+trainlabel_path+"/Sheurle_6axle_hojoon.txt","a+")
                        #trainlabel_element.write(self.util.imagefile+" 1 \n")
                        #trainlabel_element.write(self.util.imagefile.split(".png")[0]+"_1.png"+" 1 \n")
                        #trainlabel_element.close()
                        truecount+=1
                        print("True : "+str(truecount)+" False : "+str(falsecount))
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
                        print(self.util.imagefile,"False")
                        print(self.util.imagefile.split(".png")[0]+"_1.png","False")
                        #trainlabel_element=open("./"+trainlabel_path+"/Sheurle_6axle_hojoon.txt","a+")
                        #trainlabel_element.write(self.util.imagefile+" 0 \n")
                        #trainlabel_element.write(self.util.imagefile.split(".png")[0]+"_1.png"+" 0 \n")
                        #trainlabel_element.close()
                        falsecount+=1
                        print("True : "+str(truecount)+" False : "+str(falsecount))
                        time.sleep(0.2)
                        self.done=True
                reward = train.step(action,car)
                nextvalid,rear_count=car.update(dt,self.util.image,type=self.vehicle,rear_count=rear_count)

                same_check_list.append(np.array(car.position))
                if (nextvalid!=1 and nextvalid!=0):
                    self.done=True
                if nextvalid==0:
                    #print("Collision!!!")
                    reward=-0.4
                elif nextvalid==2:
                    print(self.util.imagefile,"True")
                    print(self.util.imagefile.split(".png")[0]+"_1.png","True")
                    #trainlabel_element=open("./"+trainlabel_path+"/Sheurle_6axle_hojoon.txt","a+")
                    #trainlabel_element.write(self.util.imagefile+" 1 \n")
                    #trainlabel_element.write(self.util.imagefile.split(".png")[0]+"_1.png"+" 1 \n")
                    #trainlabel_element.close()
                    truecount+=1
                    print("True : "+str(truecount)+" False : "+str(falsecount))

                # Current State by Image
                next_state=self.util.get_instant_image(car.position,car.angle,car.carwidth,car.carlength,self.scope_image_size,self.scope_image_resize)
                next_state=next_state.flatten()
                if FLAGS.frame_size > 1:
                    state_with_frame.append(next_state)
                    next_state = np.array(state_with_frame)
                    next_state = np.reshape(next_state, (1, train.RAM_FIXED_LENGTH, FLAGS.frame_size))
                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, self.done))

                state = next_state
                e_reward += reward
                step_count += 1
                # Drawing
                self.screen.fill((0, 0, 0))
                rotated = pygame.transform.rotate(car_image, car.angle)
                rect = rotated.get_rect()
                self.screen.blit(road_image,(0,0))
                #if [pygame.transform.rotate(stack_image,car.angle),car.position,car.angle] not in stack_list:
                #    stack_list+=[[pygame.transform.rotate(stack_image,car.angle),car.position,car.angle]]
                #for element in stack_list:
                #    self.screen.blit(element[0], element[1] * ppu - (element[0].get_rect().width / 2, element[0].get_rect().height / 2))
                #for lidar sensor
                """
                pygame.draw.aaline(self.screen, (0,0,255), [car.position[0],car.position[1]], [front_lidar[0],front_lidar[1]], 5)
                pygame.draw.circle(self.screen,(0,255,0),[int(front_lidar[0]),int(front_lidar[1])],5)
                pygame.draw.circle(self.screen,(0,255,0),[int(carfront[0]),int(carfront[1])],5)
                """
                """writing episode"""
                fontObj = pygame.font.Font('./font/times-new-roman.ttf', 30)
                textSurfaceObj = fontObj.render("Episode "+str(episode), True, (255,255,255), (0,0,0))
                textRectObj = textSurfaceObj.get_rect()
                textRectObj.center = (100,30)
                self.screen.blit(textSurfaceObj, textRectObj) 

                self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2)) 
                pygame.display.flip()
                #print(time.time()-timemarker,len(result))
                #self.clock.tick(self.ticks)
                global_step+=1
if __name__ == '__main__':
    game = Game()
    game.run()
