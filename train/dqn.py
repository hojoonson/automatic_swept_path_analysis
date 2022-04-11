'''DQN Class
DQN(NIPS-2013)
'Playing Atari with Deep Reinforcement Learning'
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
'Human-level control through deep reinforcement learning'
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
'''
import numpy as np
import tensorflow as tf

from model import MLPv1, Custom_CNN_forimage_v1, Custom_CNN_forimage_v2, Custom_CNN_forimage_v3
from utility import Vehicle

class DeepQNetwork:

    def __init__(self, session: tf.compat.v1.Session, model_name: str, input_size: int, output_size: int,
                 learning_rate: float = 0.0001, frame_size: int = 1, name: str = 'main') -> None:
        '''DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        '''
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.frame_size = frame_size

        self.net_name = name
        self.learning_rate = learning_rate

        self._build_network(model_name=model_name)

    def _build_network(self, model_name='MLPv1') -> None:
        with tf.compat.v1.variable_scope(self.net_name):
            if self.frame_size > 1:
                X_shape = [None] + list(self.input_size) + [self.frame_size]
            else:
                X_shape = [None] + list(self.input_size)
            self._X = tf.compat.v1.placeholder(
                tf.float32, X_shape, name='input_x')

            models = {
                'MLPv1': MLPv1,
                'Custom_CNN_forimage_v1': Custom_CNN_forimage_v1,
                'Custom_CNN_forimage_v2': Custom_CNN_forimage_v2,
                'Custom_CNN_forimage_v3': Custom_CNN_forimage_v3,
            }

            model = models[model_name](self._X, self.output_size,
                                       frame_size=self.frame_size, learning_rate=self.learning_rate)
            model.build_network()
            self._Qpred = model.inference
            self._Y = model.Y
            self._loss = model.loss
            self._train = model.optimizer

    def predict(self, state: np.ndarray) -> np.ndarray:
        '''Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        '''

        if self.frame_size > 1:
            x_shape = [-1] + list(self.input_size) + [self.frame_size]
        else:
            x_shape = [-1] + list(self.input_size)
        x = np.reshape(state, x_shape)
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        '''Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        '''

        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)

def update_action_and_get_reward(action:int, vehicle:Vehicle) -> float:
    reward=0
    if vehicle.vehicle=='car':
        car_steering=vehicle.car_steering
        if vehicle.rearvalid == 0:
            if action == 0:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = 0
                reward = 0.1
            elif action == 1:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = car_steering
                reward = 0.1
            elif action == 2:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = -car_steering
                reward = 0.1
            elif action == 3:
                reward = -0.2
            elif action == 4:
                reward = -0.2
            elif action == 5:
                reward = -0.2
        if vehicle.rearvalid == 1:
            if action == 0:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = 0
                reward = -0.05
            elif action == 1:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = car_steering
                reward = -0.05
            elif action == 2:
                vehicle.velocity.y = vehicle.car_velocity
                vehicle.steering = -car_steering
                reward = -0.05
            elif action == 3:
                vehicle.velocity.y = -vehicle.car_velocity
                vehicle.steering = 0
                reward = -0.1
            elif action == 4:
                vehicle.velocity.y = -vehicle.car_velocity
                vehicle.steering = car_steering
                reward = 0.1
            elif action == 5:
                vehicle.velocity.y = -vehicle.car_velocity
                vehicle.steering = -car_steering
                reward = 0.1

    elif vehicle.vehicle=='spmt':
        if vehicle.rearvalid==0:
            if action==0:
                vehicle.velocity.x=0
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering = 0
                reward=0.1
            elif action==1:
                vehicle.velocity.x=0
                vehicle.velocity.y=vehicle.car_velocity
                vehicle.steering = 1
                reward=0.1
            elif action==2:
                vehicle.velocity.x=0
                vehicle.velocity.y=vehicle.car_velocity
                vehicle.steering = -1
                reward=0.1
            elif action==3:
                vehicle.velocity.x=-vehicle.car_velocity
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
            elif action==4:
                vehicle.velocity.x= vehicle.car_velocity
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
            elif action==5:
                reward=-0.2
            elif action==6:
                reward=-0.2
            elif action==7:
                reward=-0.2
            elif action==8:
                reward=-0.2

        elif vehicle.rearvalid==1:      
            if action==0:
                vehicle.velocity.x=0
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering = 0
                reward=0.1
            elif action==1:
                vehicle.velocity.x=0
                vehicle.velocity.y=vehicle.car_velocity
                vehicle.steering = 1
                reward=0.1
            elif action==2:
                vehicle.velocity.x=0
                vehicle.velocity.y=vehicle.car_velocity
                vehicle.steering = -1
                reward=0.1
            elif action==3:
                vehicle.velocity.x=-vehicle.car_velocity
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
            elif action==4:
                vehicle.velocity.x= vehicle.car_velocity
                vehicle.velocity.y= vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
            elif action==5:
                vehicle.velocity.x=0
                vehicle.velocity.y=-vehicle.car_velocity
                vehicle.steering = 1
                reward=0.1
            elif action==6:
                vehicle.velocity.x=0
                vehicle.velocity.y=-vehicle.car_velocity
                vehicle.steering = -1
                reward=0.1
            elif action==7:
                vehicle.velocity.x=-vehicle.car_velocity
                vehicle.velocity.y=-vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
            elif action==8:
                vehicle.velocity.x= vehicle.car_velocity
                vehicle.velocity.y=-vehicle.car_velocity
                vehicle.steering=0
                reward=0.1
    return reward