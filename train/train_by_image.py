from utility import *
from collections import deque
from dqn import DeepQNetwork
import tensorflow as tf
import logging
from typing import List
import random
import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 3000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 15000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 32, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 4, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'Hojoon_Custom_CNN_forimage_v2', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
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
        if car.vehicle=="car":
            car_steering=36.91
            if action==0:
                car.velocity.y= car.car_velocity
                car.steering = 0
                reward=0.1
            elif action==1:
                car.velocity.y = car.car_velocity
                car.steering = car_steering
                reward=0.1
            elif action==2:
                car.velocity.y= car.car_velocity
                car.steering = -car_steering
                reward=0.1

        elif car.vehicle=="spmt":
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
        labels="scherule_trainlabels.txt"
        label_path="./trainlabels/"+labels

        f = open(label_path,"r")
        pathdata = f.readlines()
        path_list=[]
        for element in pathdata:
            if int(element.split(" ")[1])==1:
                path_list+=[element.split(" ")[0]]
        f.close()
        print(path_list)
        self.roadimage_path=path_list

        labels="scherule_testlabels.txt"
        label_path="./testlabels/"+labels

        f = open(label_path,"r")
        pathdata = f.readlines()
        path_list=[]
        for element in pathdata:
            if int(element.split(" ")[1])==1:
                path_list+=[element.split(" ")[0]]
        f.close()
        print(path_list)
        self.roadimage_path+=path_list

        self.util=utility()
        self.random_candidate=3
        self.map_updatecount=3
        self.util.imagefile=self.roadimage_path[0]                    
        pygame.init()
        pygame.display.set_caption("Swept Path Analysis")
        self.startx=296.5
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
        self.outputsize=9
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
        lossresult_path = "Sheurle_6axle_2x5_Swept_Path_Analysis_byimage_output:"+str(self.outputsize) + "_f" + str(FLAGS.frame_size) +"_transport_"+str(self.vehicle)+"_model_" +FLAGS.model_name+"_"+FLAGS.checkpoint_path + "global_step"
        if not os.path.exists(lossresult_path):
            os.makedirs(lossresult_path)
        lossresult=open("./"+lossresult_path+"/loss.txt","w+")
        rewardresult_path = "Sheurle_6axle_2x5_Swept_Path_Analysis_byimage_output:"+str(self.outputsize) + "_f" + str(FLAGS.frame_size) +"_transport_"+str(self.vehicle)+"_model_" +FLAGS.model_name+"_"+FLAGS.checkpoint_path + "global_step"
        if not os.path.exists(rewardresult_path):
            os.makedirs(rewardresult_path)
        rewardresult=open("./"+rewardresult_path+"/reward.txt","w+")
        with tf.Session() as sess:
            mainDQN = DeepQNetwork(sess, FLAGS.model_name, train.input_size, train.output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
            targetDQN = DeepQNetwork(sess,FLAGS.model_name, train.input_size, train.output_size, frame_size=FLAGS.frame_size, name="target")

            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            #saver.restore(sess, tf.train.latest_checkpoint("Swept_Path_Analysis_byimage_output_10_f4_transport_spmt_model_Hojoon_Custom_CNN_forimage_v2_checkpoint"))

            # initial copy q_net -> target_net
            copy_ops = train.get_copy_var_ops(dest_scope_name="target",
                                        src_scope_name="main")
            sess.run(copy_ops)
            
            global_step = 1
            stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position,car.angle]]
            for episode in range(FLAGS.max_episode_count):
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
                        dt = 0.03
                    elif self.vehicle=="spmt":
                        dt = 0.04
                    if np.random.rand() < e:
                        # random action
                        action = train.action_sample(car.vehicle,self.outputsize)
                        #print("random action",action)
                    else:
                        # Get new state and reward from environment
                        action = np.argmax(mainDQN.predict(state))
                        #print("Q-func action",action)
                    

                    reward = train.step(action,car)
                    nextvalid,rear_count=car.update(dt,self.util.image,type=self.vehicle,rear_count=rear_count)
                    """
                    front_lidar=self.util.front_lidar(car.position,car.angle,car.carlength,100)
                    front_lidar=np.array(front_lidar[0])
                    carfront=np.array([car.position[0]-car.carlength/2*math.sin(radians(car.angle)),car.position[1]-car.carlength/2*math.cos(radians(car.angle))])
                    reward-=(0.5-np.linalg.norm(front_lidar-carfront)*0.5/100)
                    """
                    event=pygame.event.get()
            
                    if len(event)!=0:
                        if event[0].type==pygame.KEYDOWN:
                            if event[0].key==pygame.K_q:
                                nextvalid=3
                    same_check_list.append(np.array(car.position))
                    if nextvalid==0:
                        #print("Collision!!!")
                        reward=-4.0
                    if nextvalid==2:
                        print("Finish the Analysis!!!")
                        reward=0.4
                    if nextvalid==3:
                        print("force quit to next episode")
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
                            print(" - step_count : "+str(step_count)+", reward: "+str(e_reward)+" ,loss: "+str(loss))
                        if global_step % 100 == 0:
                            lossresult.write("global_step:"+str(global_step)+ " loss: "+str(loss)+"\n")
                    if step_count % FLAGS.target_update_count == 0:
                        sess.run(copy_ops)

                    state = next_state
                    e_reward += reward
                    step_count += 1

                    if len(same_check_list)==50 and len(np.unique(same_check_list))==2:
                        rewardresult.write("episode:"+str(episode)+ " reward: "+str(e_reward)+"\n")
                        self.done=True
                    if (nextvalid!=1 and nextvalid!=0) or (step_count!=0 and step_count%1000==0):
                        rewardresult.write("episode:"+str(episode)+ " reward: "+str(e_reward)+"\n")
                        self.done=True

                    # save model checkpoint
                    if global_step % FLAGS.save_step_count == 0:
                        checkpoint_path = "Sheurle_6axle_2x5_Swept_Path_Analysis_byimage_output:"+str(self.outputsize) + "_f" + str(FLAGS.frame_size) +"_transport_"+str(self.vehicle)+"_model_" +FLAGS.model_name+"_"+FLAGS.checkpoint_path + "global_step"
                        if not os.path.exists(checkpoint_path):
                            os.makedirs(checkpoint_path)

                        saver.save(sess, checkpoint_path, global_step=global_step)
                        logger.info("save model for global_step: "+str(global_step))

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
