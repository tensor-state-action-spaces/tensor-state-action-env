"""A very simple version of grid soccer"""
# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from gym.core import Env

from gym.spaces import Discrete, Box
from gym.spaces import Dict
from gym.utils import seeding

from .utils import socceragent
import pygame
import numpy as np
import math
import os
from PIL import Image

green = 0, 255, 0
red = 255, 0, 0
blue = 0, 0, 255
black = 0, 0, 0
yellow = 255, 255, 0
scale = 20
white = 255, 255, 255

"""
    Time of passing is ignored in this grid soccer domain.
"""

class GridSoccerPassing(Env):
    def __init__(self,
                 length=10,
                 width=10,
                 num_agents=(5, 4),
                 max_steps=500,
                 scale=20,
                 masking=False,
                 mask_reward=False,
                 use_state_feature=False,
                 use_tensor_action=True,
                 ground_truth_mask=False):
        """
            Initialize the environments.

            length x width is the size for the field
            length --x axis, width --y axis
            num_agents record the numbers for both teams

            TODO: take chip kick into account
        """
        self.length = length
        self.width = width
        self.num_agents = num_agents
        self.scale = scale

        self.steps = 0
        #self.chip_kick = False  # noqa: E265        
        self.soccerfield = socceragent.SoccerField(self.length, self.width,self.scale,
                                                   self.num_agents)
        self.max_steps = max_steps
        self.masking = masking
        self.mask_reward = mask_reward
        
        self.use_tensor_action = use_tensor_action
        self.use_state_feature = use_state_feature
        
        self.ground_truth_mask = ground_truth_mask
        """
            Action space:
                "teamA":
                "teamB":
                "passing": True, False

        """
        self.teamA_names = ["agent%02d" % (i) for i in range(num_agents[0])]
        self.teamB_names = ["agent%02d" % (i) for i in range(num_agents[1])]

        self.action_teamA = Dict(
            {agent: Discrete(5)
             for agent in self.teamA_names})
        self.action_teamB = Dict(
            {agent: Discrete(5)
             for agent in self.teamB_names})

        self.action_space = Discrete(self.length * self.width)

        if self.use_state_feature is True:
            self.observation_space = Box(0, 100, shape=(3 * (self.num_agents[0] + self.num_agents[1] + 1),))
        elif self.masking is True:
            self.observation_space = Box(
                0,
                255,
                shape=(self.length * self.scale, self.width * self.scale, 2))
        else:
            self.observation_space = Box(
                0,
                255,
                shape=(self.length * self.scale, self.width * self.scale, 1))

        self.obs = None
        """
            "teamA": locations of agents in team A
            "teamB": locations of agents in team B
            "the agent with the ball": the agent number in team A with the ball
            "target cell": Could be just random unoccupied cell, or the cell of
                           a teammate
        """
        if self.use_tensor_action is True:
            self.action_space = Discrete(self.length * self.width)
        else:
            # Action would be the agent's index, including teammates
            # and opponents
            self.action_space = Discrete(num_agents[0])

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
            reset the environment.

            Default mode is random, which means that it will
            randomly choose a target point

            Locations of every agent is randomly given

            self.obs is the self.obs of the envionment,
            It should contain several contents:
                self.obs[0] -Locations of every agent
                self.obs[1] -The agent that carries the ball
                self.obs[2] -Location of the target cell
        """

        self.obs = self.soccerfield.reset()
        self.draw()
        self.steps = 0
        self._first_passing_opportunity_time = None

        if self.use_state_feature is False:
            return self.obs['image']
        else:
            return self.rtr_state_feature(self.obs)

    def reward(self, debug_info):
        done = False

        flag = debug_info["flag"]
        intercept = debug_info["intercept"]
        reward = 0
        if self.mask_reward is False:
            reward = -1
            if flag == socceragent.PassStatus.PASS_IN_ROW_OR_COLUMN:
                if intercept:
                    reward = -10
                else:
                    reward = 10
                done = True
            elif flag == socceragent.PassStatus.PASS_NOT_TO_TEAMMATE or flag == socceragent.PassStatus.PASS_NOT_IN_ROW_OR_COLUMN:
                reward = -10
                done = True
        else:
            if flag ==socceragent.PassStatus.PASS_IN_ROW_OR_COLUMN and not intercept:
                reward = 0
                done = True
            else:
                reward = -10
                done = True
            if flag ==socceragent.PassStatus.HOLDBALL:
                reward = 0
                done = False

        return reward, done

    def rtr_act(self, action):
        if self.use_tensor_action is True:
            return action
        else:
            if action < self.num_agents[0]:
                x = self.obs["teamA"][action][0]
                y = self.obs["teamA"][action][1]
            else:
                x = self.obs["teamB"][action-self.num_agents[0]][0]
                y = self.obs["teamB"][action-self.num_agents[0]][1]
            return x * self.width + y

    def step(self, passing):
        """
            Step function

            If an agent's action hits another agent or hits the boundary,
            it would stay still.
        """
        # if self._first_passing_opportunity_time is None:
        #     if self._can_pass():
        #         self._first_passing_opportunity_time = self.steps

        passing = self.rtr_act(passing)
        self.action_teamA = {
            agent: np.random.randint(5)
            for agent in self.teamA_names
        }
        self.action_teamB = {
            agent: np.random.randint(5)
            for agent in self.teamB_names
        }

        action = dict()
        action["teamA"] = self.action_teamA
        action["teamB"] = self.action_teamB
        action["passing"] = [
            math.floor(passing / self.width), passing % self.width
        ]

        self.obs, debug_info = self.soccerfield.step(action)

        reward, done = self.reward(debug_info)

        self.steps = self.steps + 1

        if self.steps >= self.max_steps:
            done = True
        self.draw()
        if self.use_state_feature is False:
            return self.obs['image'], reward, done, {
                'intercepted':
                debug_info["intercept"]
                # 'first_passing_opportunity_time':
                # self._first_passing_opportunity_time
            }
        else:
            return self.rtr_state_feature(self.obs), reward, done, {
                'intercepted': debug_info["intercept"]}

    def rtr_state_feature(self, obs):
        """
        If we are using state feature, the representation should be as full as full view of image.
        each agent would be location x, y + index
        each opponent would be location x, y + index 
        
        """
        rtr_obs = []
        rtr_obs = self.obs["teamA"][self.obs["ball_carrier"]] + [self.obs["ball_carrier"]]
        i = 0
        for loc in self.obs["teamA"]:
            rtr_obs = rtr_obs + loc + [i]
            i = i + 1
        for loc in self.obs["teamB"]:
            rtr_obs = rtr_obs + loc + [i]
            i = i + 1
        return rtr_obs
    
    def draw(self):
        size = self.length * self.scale, self.width * self.scale
        action_mask = np.ndarray(shape=(size[0], size[1], 1), dtype=np.int32)
        action_mask[:, :, 0] = 0
        ball_carrier = self.obs["ball_carrier"]

        if self.ground_truth_mask is True:
            x_min = self.obs["teamA"][ball_carrier][0] - 1
            x_max = self.obs["teamA"][ball_carrier][0] + 1
            y_min = self.obs["teamA"][ball_carrier][1] - 1
            y_max = self.obs["teamA"][ball_carrier][1] + 1

            while ([x_min, self.obs["teamA"][ball_carrier][1]] not in self.soccerfield.agent_locations and x_min >=0):
                x_min -= 1

            if [x_min, self.obs["teamA"][ball_carrier][1]] in self.soccerfield.agent_locations[0:self.num_agents[0]]:
                x_min -= 1

            x_min += 1

            while (x_max < self.length and [x_max, self.obs["teamA"][ball_carrier][1]] not in self.soccerfield.agent_locations):
                x_max += 1

            if [x_max, self.obs["teamA"][ball_carrier][1]] in self.soccerfield.agent_locations[0:self.num_agents[0]]:
                x_max +=1

            while ([self.obs["teamA"][ball_carrier][0], y_min] not in self.soccerfield.agent_locations and y_min >=0):
                y_min -= 1

            if [self.obs["teamA"][ball_carrier][0], y_min] in self.soccerfield.agent_locations[0:self.num_agents[0]]:
                y_min -= 1

            y_min += 1

            while (y_max < self.width and [self.obs["teamA"][ball_carrier][0], y_max] not in self.soccerfield.agent_locations):
                y_max +=1

            if [self.obs["teamA"][ball_carrier][0], y_max] in self.soccerfield.agent_locations[0:self.num_agents[0]]:
                y_max += 1

        x = self.obs["teamA"][ball_carrier][0] * self.scale
        y = self.obs["teamA"][ball_carrier][1] * self.scale

        if self.ground_truth_mask is True:
            action_mask[x:x + self.scale, (y_min)*self.scale:y_max*self.scale, 0] = 255
            action_mask[(x_min)*self.scale:x_max*self.scale, y:y + self.scale, 0] = 255
        else:
            action_mask[x:x + self.scale, :, 0] = 255
            action_mask[:, y:y + self.scale, 0] = 255

        try:
            screen = pygame.display.set_mode(size, 0, 32)

            pygame.display.set_caption("GridSoccer")

        except:
            os.environ["SDL_VIDEODRIVER"] = 'dummy'
            screen = pygame.display.set_mode(size, 0, 32)
            pygame.display.set_caption("GridSoccer")

        screen.fill(green)

        for location in self.obs["teamA"]:
            pygame.draw.rect(screen, blue, [
                location[0] * self.scale, location[1] * self.scale,
                self.scale, self.scale
            ])

        for location in self.obs["teamB"]:

            pygame.draw.rect(screen, yellow, [
                location[0] * self.scale, location[1] * self.scale,
                self.scale, self.scale
            ])

        pygame.draw.rect(screen, red, [
            self.obs["teamA"][ball_carrier][0] * self.scale,
            self.obs["teamA"][ball_carrier][1] * self.scale, self.scale,
            self.scale
        ])
        img = pygame.surfarray.array3d(screen)
        img = Image.fromarray(img).convert('L')
        img = np.array(img)[...,np.newaxis]

        self.obs["image"] = img
        if self.masking is True:
            self.obs["image"] = np.concatenate(
                (self.obs["image"], action_mask), axis=2)

        return

    def render(self, close=False):
        """
            rendering of the environment
            blue for team A
            yellow for team B
        """
        self.draw()
        """
            When the display mode does not work, such as running on a cluster.
        """
        try:
            pygame.display.update()
        except:
            return



    def _can_pass(self):
        # find the agent with the ball
        agent_with_ball = self.soccerfield.teamA[self.soccerfield.ball]

        for agent in self.soccerfield.teamA:
            # skip if this is the agent with the ball
            if agent == agent_with_ball:
                continue
            # is this agent in the same row or column as the one with
            # the ball?
            if (agent_with_ball.loc_x != agent.loc_x
                    and agent_with_ball.loc_y != agent.loc_y):
                continue
            # if the same column, verify that there are no
            # interceptions
            if agent_with_ball.loc_x == agent.loc_x:
                if agent_with_ball.loc_y < agent.loc_y:
                    (agent_with_smaller_y,
                     agent_with_bigger_y) = (agent_with_ball, agent)
                else:
                    (agent_with_smaller_y,
                     agent_with_bigger_y) = (agent, agent_with_ball)
                for opponent in self.soccerfield.teamB:
                    if (opponent.loc_x == agent_with_ball.loc_x
                            and opponent.loc_y > agent_with_smaller_y.loc_y
                            and opponent.loc_y < agent_with_bigger_y.loc_y):
                        # opponent is in the same column and is in
                        # between the agent and this teammate
                        continue

                # Either there is only empty space between the agent
                # and the teammate, or there is another teammate
                # between the agent and this teammate. In either case
                # a successful pass could be made
                return True
            elif agent_with_ball.loc_y == agent.loc_y:
                # if the same row, verify no interceptions
                if agent_with_ball.loc_x < agent.loc_x:
                    (agent_with_smaller_x,
                     agent_with_bigger_x) = (agent_with_ball, agent)
                else:
                    (agent_with_smaller_x,
                     agent_with_bigger_x) = (agent, agent_with_ball)
                for opponent in self.soccerfield.teamB:
                    if (opponent.loc_y == agent_with_ball.loc_y
                            and opponent.loc_x > agent_with_smaller_x.loc_x
                            and opponent.loc_x < agent_with_bigger_x.loc_x):
                        # opponent is in the same column and is in
                        # between the agent and this teammate
                        continue

                # Either there is only empty space between the agent
                # and the teammate, or there is another teammate
                # between the agent and this teammate. In either case
                # a successful pass could be made
                return True

        return False

        
