"""Gym environ for customized breakout"""
# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from gym.spaces import Discrete, Box
from gym.utils import seeding
from gym.spaces import Dict
from .utils import engine
from .utils import util
import numpy as np
import math
import os

class BreakOut():
    def __init__(self,
                 height=84,
                 width=84,
                 channel=4,
                 history=3,
                 evaluate=False,
                 use_masking=True,
                 use_state_feature=False):

        self.height = height
        self.width = width
        self.channel = channel
        self.history = history
        self.use_masking = use_masking
        self.use_state_feature = use_state_feature
        
        self.world = self.get_world()
                
        if self.use_masking is True:
            self.channel = self.history + 1
            print(util.font_colors.OKGREEN+"CURRENTLY USING MASKING!!!"+util.font_colors.ENDC)
        else:
            self.channel = self.history
            print(util.font_colors.WARNING+"NOT USING MASKING!!!"+util.font_colors.ENDC)
        
        
        self.num_paddles=1
        self.action_space = Discrete(self.width * self.height)

        self.evaluate = evaluate
        self.use_masking = use_masking
        self.steps = 0

        self._engine = engine.PyGameEngine('BreakOut-Fcn-v0', self.get_world())
        self._engine.update(self.world)

        if self.use_state_feature is False:
            self.observation_space = Box(low=0, high=255, shape=(self.width, self.height, self.channel))
        else:
            # ball, paddle, y_ofs , all the states of the brick
            self.observation_space = Box(low=0,high=max(self.height, self.width), shape=(2 + 2 + 1 + self._engine.num_bricks,))

        
        
    def get_obs(self):
        return self._engine.get_obs(use_state_feature=self.use_state_feature)

    def get_world(self):
        """
        Information in the world:
        -width
        -length
        """
        world = dict()
        world["width"] = self.width
        world["height"] = self.height
        world["history"] = self.history
        world["masking"] = self.use_masking
        return world

    def get_img(self):
        return self._engine.get_img()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def set(self, world, evaluate=False):
    #     """
    #     This set function specifically is designed for chaning window size
    #     """
    #     self.world = world
    #     self.height = world["height"]
    #     self.width = world["width"]
    #     self.history = world["history"]
    #     self.use_masking = world["masking"]

    #     if self.use_masking is True:
    #         self.channel = self.history + 1
    #     else:
    #         self.channel = self.history
    #         print("NOT USING MASKING!!!")
    #     self.evaluate = evaluate

    #     self.action_space = Discrete(self.width * self.height)
    #     self.observation_space = Box(low=0, high=255, shape=(self.width, self.height, self.channel))
    #     self._engine.update(self.world)

    def reset(self):
        self.steps = 0
        self._engine.init_game()
        return self.get_obs()

    def step(self, action):
        """
        action: a coordination of a pixel
        """
        self.steps += 1
        self._engine.step(self.Wrapper(action))
        obs = self.get_obs()
        reward = self._reward()
        if self.evaluate is True:
            if reward <0:
                reward = 0
        done  = self._engine.get_state()
        if self.steps > 1000:
            done = True

        debug_info = dict()
        debug_info["valid"] = self._engine.valid
        """
        if self.steps <30:
            done = False
        else:
            done = True
        """
        return obs, reward, done, debug_info

    def get_action_masking(self):
        """Currently return action masing directly. Could return one hot masking for hard masking later"""
        return self._engine.action_masking()
    
    def Wrapper(self,action):
        """
        This function changes action to valid actions for engine
        Input action: pixel-based control
        Return: +- x pixels
        """
        # print(self._engine.valid_action([math.floor(action / self.height), action % self.height]))
        return self._engine.valid_action([math.floor(action / self.height), action % self.height]) 
            
    def paddle_left(self):
        return self._engine.paddle_left()

    def paddle_middle(self):
        return self._engine.paddle_middle()

    def _reward(self):
        """
        Reward Setting
        """
        return self._engine.reward()
    
    def render(self):
        self._engine.update(self.get_world())
        self._engine.render()

    def record(self, save_dir, save_filename):
        save_file = os.path.join(save_dir, save_filename)
        self._engine.record(save_file)

class BreakOutDiscrete(BreakOut):
    metadata={'render.modes':['human']}

    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)

    def reset(self):
        return super().reset()

    def set(self, world, evaluate=False):
        super().set(world, evaluate)
        self.action_space = Discrete(self._engine.max_horizontal_move*2+1)
    
    def step(self, action):
        """
        Action: Discrete action, 0 for left, 1 for middle, 2 for right
        """
        action -= self._engine.max_horizontal_move
        paddle_pos = self._engine.paddle_middle()
        converted_action = (paddle_pos[0] + action) * self.height + paddle_pos[1]
        obs, reward, done, _ = super()._step(converted_action)
        return obs, reward, done, {}
