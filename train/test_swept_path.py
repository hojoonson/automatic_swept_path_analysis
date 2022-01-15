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
flags.DEFINE_string('test_model','./result_log/7_19095_result_converge_justfront','model path')
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 128, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', '1', 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'Hojoon_Custom_MLPv0', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 1, 'Learning rate. ')
flags.DEFINE_boolean('step_verbose', True, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
FLAGS = flags.FLAGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Test:
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.RAM_FIXED_LENGTH=input_size[0]
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

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder
    def step(self,action,car):
        reward=0
        if car.vehicle=="car":
            if action==0:
                car.velocity.y= car.car_velocity
            elif action==1:
                car.velocity.y = car.car_velocity
                car.steering = 30
            elif action==2:
                car.velocity.y= car.car_velocity
                car.steering = -30
            
            elif action==3:
                car.velocity.y= -car.car_velocity
            elif action==4:
                car.velocity.y= -car.car_velocity   
                car.steering = 30
            elif action==5:
                car.velocity.y= -car.car_velocity
                car.steering = -30
            return reward

    def action_sample(self,vehicle):
        if vehicle=="car":
            return random.randint(0,2)

class Game:
    def __init__(self):
        self.roadimage_path=glob.glob("./testimages/*.png")
        self.util=utility()
        self.util.imagefile=self.roadimage_path[0]                    
        pygame.init()
        pygame.display.set_caption("Swept Path Analysis")
        self.num_of_points=16
        self.startx=300
        self.starty=500
        self.startangle=0
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height), HWSURFACE | DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.ticks = 1000
        self.vehicle="car"
        self.outputsize=3

    
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
            test=Test(input_size=(self.num_of_points*2,),output_size=self.outputsize)
        elif car.vehicle=="spmt":
            test=Test(input_size=(self.num_of_points*2,),output_size=self.outputsize)
        logger.info("FLAGS configure.")
        #logger.info(FLAGS.__flags)
        # store the previous observations in replay memory
        
        with tf.Session() as sess:
            mainDQN = DeepQNetwork(sess, FLAGS.model_name, test.input_size, test.output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.test_model))

            # initial copy q_net -> target_net
            copy_ops = test.get_copy_var_ops(dest_scope_name="target",
                                        src_scope_name="main")
            sess.run(copy_ops)
            
            global_step = 1
            for episode in range(FLAGS.max_episode_count):
                #set road image!
                self.util.imagefile=self.roadimage_path[episode%len(self.roadimage_path)]
                self.util.image=cv2.imread(self.util.imagefile)
                self.util.edge=cv2.Laplacian(self.util.image,cv2.CV_8U)
                self.util.edge=cv2.cvtColor(self.util.edge,cv2.COLOR_BGR2GRAY)
                road_image=pygame.image.load(self.util.imagefile)
                
                #reset collision, finish valid
                self.collision_valid=0
                self.finish_valid=0
                # Initializing
                car = Car(x=self.startx,y=self.starty,angle=self.startangle,vehicle=self.vehicle)
                nextvalid=1
                stack_list=[[pygame.transform.rotate(stack_image,car.angle),car.position]]
                self.done = False

                # Current State by Lidar Sensor
                result=self.util.lidar_sensor(car.position,car.angle,num_of_points=self.num_of_points)
                state=np.array(result)-np.array(car.position)
                state=state.flatten()
                
                if FLAGS.frame_size > 1:
                    state_with_frame = deque(maxlen=FLAGS.frame_size)

                    for _ in range(FLAGS.frame_size):
                        state_with_frame.append(state)
                    state = np.array(state_with_frame)
                    state = np.reshape(state, (1, test.RAM_FIXED_LENGTH, FLAGS.frame_size))

                while not self.done:
                    #dt = self.clock.get_time() / 1000
                    dt = 0.04
                    action = np.argmax(mainDQN.predict(state))
                    # Get new state and reward from environment
                    test.step(action,car)
                    nextvalid=car.update(dt,self.util.image,car.vehicle)
                    event=pygame.event.get()
                    if len(event)!=0:
                        if event[0].type==pygame.KEYDOWN:
                            if event[0].key==pygame.K_q:
                                nextvalid=3
                    if nextvalid!=1:
                        self.done=True
                    if nextvalid==0:
                        print("Collision!!!")
                    elif nextvalid==2:
                        print("Finish the Analysis!!!")
                    elif nextvalid==3:
                        print("force quit to next episode")

                    # Current State by Lidar Sensor
                    result=self.util.lidar_sensor(car.position,car.angle,num_of_points=self.num_of_points)
                    next_state=np.array(result)-np.array(car.position)
                    next_state=next_state.flatten()
                    if FLAGS.frame_size > 1:
                        state_with_frame.append(next_state)

                        next_state = np.array(state_with_frame)
                        next_state = np.reshape(next_state, (1, test.RAM_FIXED_LENGTH, FLAGS.frame_size))
                    state = next_state
                    # Drawing
                    self.screen.fill((0, 0, 0))
                    rotated = pygame.transform.rotate(car_image, car.angle)
                    rect = rotated.get_rect()
                    self.screen.blit(road_image,(0,0))
                    if pygame.transform.rotate(stack_image,car.angle)!=stack_list[-1][0] and\
                        [pygame.transform.rotate(stack_image,car.angle),car.position] not in stack_list:
                        stack_list+=[[pygame.transform.rotate(stack_image,car.angle),car.position]]
                    for element in stack_list:
                        self.screen.blit(element[0], element[1] * ppu - (element[0].get_rect().width / 2, element[0].get_rect().height / 2))
                    
                    for element in result:
                        pygame.draw.aaline(self.screen, (0,0,255), [car.position[0],car.position[1]], [element[0],element[1]], 5)
                        pygame.draw.circle(self.screen,(0,255,0),[int(element[0]),int(element[1])],5)
                        
                        """writing point"""
                        
                        fontObj = pygame.font.Font('./font/times-new-roman.ttf', 30)
                        textSurfaceObj = fontObj.render("Episode "+str(episode), True, (255,255,255), (0,0,0))
                        textRectObj = textSurfaceObj.get_rect()
                        textRectObj.center = (100,30)
                        self.screen.blit(textSurfaceObj, textRectObj) 
                        
                    self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2)) 
                    pygame.display.flip()
                    #print(time.time()-timemarker,len(result))
                    self.clock.tick(self.ticks)
                    global_step+=1
            pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
