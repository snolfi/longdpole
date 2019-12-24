"""
Classic long double-pole balancing problem system,
see: Pagliuca P., Milano N. and Nolfi S. (2018). Maximizing adaptive power in neuroevolution. PLoS ONE 13(7): e0198788.
Variation of the classic cart-pole system implemented by Rich Sutton et al.
"""
import os
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

#import time

import ErDpole

class LongdpoleEnv(gym.Env):
    """
    Description:
        Two poles of different length and mass are attached by two un-actuated joints to a cart, which moves along a frictionless track.
        The poles starts upright, and the goal is to prevent them from falling over by increasing and reducing the cart's velocity.
    Observation:
        Type: Box(3)
        Num	Observation
        0	Cart Position  Min=0.5,      Max=0.5         corresponding [-2.4, 2.4] m
        2	Pole1 Angle    Min=1.208325  Max=1.208325    corresponding [-36, 36] degrees
        3	Pole2 Angle    Min=1.208325  Max=1.208325    corresponding [-36, 36] degrees
		
    Actions:
        Type: Box(1)
		0	Torque        Min=-1.0,      Max= 1.0
		
    Reward:
        Reward is 1 for every step in which the cart and the poles are within bounds
    Starting State:
        Initialized randomly with at the start of each evaluation episode in the following ranges with uniform distribution:
        cart position   [-1.944, 1.944] m
        cart velocity   [-1.215, 1.215] m/s
        pole1 position  [-0.10472, 0.10472] rad
        pole1 velocity  [-0.135, 0.135] rad/s
        pole2 position  [-0.10472, 0.10472] rad
        pole2 velocity  [-0.135, 0.135] rad/s
		
    Episode Termination:
        Pole Angles is more than 36 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 1000
	Generalization Performance
		average rewards obtained during 500 evaluation episodes

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, render=False):
        # action encodes the torque applied by the motor of the cart pole
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype='float32')
        # observation encodes xpos of the cart and angle of the two poles in radiants
        self.observation = []
        self.observation_space = spaces.Box(np.array([-2.4, -0.628329, -0.628329]),
                np.array([-2.4, -0.628329, -0.628329]), dtype='float32')
        # make the environment
        self.env = ErDpole.PyErProblem()
        # create vector for observation, action, and done
        # and share the links with c++/cython library 
        self.ob = np.arange(self.env.ninputs, dtype=np.float32)
        self.ac = np.arange(self.env.noutputs, dtype=np.float32)
        self.done = np.arange(1, dtype=np.intc)
        self.env.copyObs(self.ob)
        self.env.copyAct(self.ac)
        self.env.copyDone(self.done)
        self.seed()
        
        self.viewer = None
        self.state = None
        self.x_threshold = 2.4
        self.length = 0.5
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.env.reset() 
        return self.ob

    def step(self, action):
        # copy action into the self.ac vector
        self.ac[0] = action[0]
        reward = self.env.step()
        return self.ob, reward, self.done, {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * 0.5)
        pole2width = 10.0
        pole2len = scale * (2 * 0.5) * self.length
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            # next 7 lines added for the second pole
            l2,r2,t2,b2 = -pole2width/2,pole2width/2,pole2len-pole2width/2,-pole2width/2
            pole2 = rendering.FilledPolygon([(l2,b2), (l2,t2), (r2,t2), (r2,b2)])
            pole2.set_color(.6,.8,.4)
            self.pole2trans = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole
            self._pole2_geom = pole2

        #if self.state is None: return None  commented by stefano

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        pole2 = self._pole2_geom
        l2,r2,t2,b2 = -pole2width/2,pole2width/2,pole2len-pole2width/2,-pole2width/2
        pole2.v = [(l2,b2), (l2,t2), (r2,t2), (r2,b2)]

        #x = self.state commented by stefano
        x = self.ob
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[1])
        self.pole2trans.set_rotation(-x[2])

        #time.sleep(1)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




