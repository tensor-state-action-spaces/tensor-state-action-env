"""Environments for Take-the-Treasure. It is a simplified version of keep-away tasks from robot soccer."""
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

class TakeTheTreasure(Env):
    def __init__(self,
                 length=10,
                 width=10,
                 num_agents=(3, 3),
                 max_steps=500,
                 scale=20,
                 masking=False,
                 mode=1,
                 extreme_mode=False,
                 moving_penalty=False,
                 wrapper=None,
                 use_state_feature=False,
                 use_tensor_action=True):
        """
            Initialize the environments.

            length x width is the size for the field
            length --x axis, width --y axis
            num_agents record the numbers for both teams

        """
        self.length = length
        self.width = width
        self.num_agents = num_agents
        self.scale = scale
        self.steps = 0
        #self.chip_kick = False  # noqa: E265
        self.soccerfield = socceragent.SoccerField(self.length, self.width,
                                                   self.num_agents)
        self.soccerfield.num_agents = num_agents        
        self.max_steps = max_steps
        self.viewer = None
        self.opponent_steps = 0
        self.blocking = False
        self.masking = masking
        self.mode = mode # mode 0: passing and moving separated
        
        self.extreme_mode = extreme_mode

        self.use_state_feature = use_state_feature
        self.use_tensor_action = use_tensor_action
        
        self.moving_penalty = moving_penalty
        self.moving_steps = 0

        self.action_mask = []
        self.teamA_names = ["agent%02d" % (i) for i in range(num_agents[0])]
        self.teamB_names = ["agent%02d" % (i) for i in range(num_agents[1])]

        self.action_teamA = Dict(
            {agent: Discrete(5)
             for agent in self.teamA_names})
        self.action_teamB = Dict(
            {agent: Discrete(5)
             for agent in self.teamB_names})

        if self.use_state_feature is True:
            self.observation_space = Box(0, 100, shape=(3 * (self.num_agents[0] + self.num_agents[1] + 1), self.num_agents[0]))

        elif self.masking is False:       
            self.observation_space = Box(0, 255, shape=(self.length * self.scale,
                                                        self.width * self.scale, 1))
        else:
            self.observation_space = Box(0, 255, shape=(self.length * self.scale,
                                                        self.width * self.scale, 1 + self.num_agents[0]))
        self.obs = None

        if self.use_tensor_action is True:
            self.action_space = Box(0, 2 * self.length * self.width-1,shape=(self.num_agents[0],), dtype=np.uint8)
        else:
            self.action_space = Box(0, self.num_agents[0] + 4,shape=(self.num_agents[0],), dtype=np.uint8)
            
        # self.action_space = Dict({
        #     "passing":
        #     Discrete(self.length * self.width),
        #     "move":
        #     self.action_teamA
        # })

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
        self.opponent_steps = 0
        self.blocking = False
        
        if self.use_state_feature is False:
            return self.Wrapper(self.obs['image'],self.obs['teamA'])
        else:
            return self.rtr_state_feature(self.obs)

    def reward(self, flag, intercept, next_one, blocked):
        done = False
        rewards = []

        if intercept or blocked:
            done = True
        """
        if flag == 1:
            print("Passing")
            print(self.steps)
        """
        if self.mode == 0:
            for i in range(self.num_agents[0]):
                if intercept or blocked:
                    reward = -1 * 1.0
                else:
                    reward = 0.0
                name = "agent%02d" % (i)
                passing_without_ball = False
                if i != self.obs["ball_carrier"]:
                    if self.action_teamA[name] < 0:
                        if self.obs["teamA"][i][0] * self.length + self.obs["teamA"][i][1] != self.action_teamA[name] + self.length * self.width:
                            passing_without_ball = True
                    if passing_without_ball is True:
                        reward += -1.0
                else:
                    if flag == 0:
                        reward += -1.0
                rewards.append(reward)
        else:

            for i in range(self.num_agents[0]):

                if intercept or blocked:
                    reward = -1.0
                else:
                    reward = 0.0
                name = "agent%02d" % (i)
                if self.extreme_mode is True and flag == 1:
                    reward = 0
                    if intercept or blocked:
                        reward = -1.0
                else:
                    if i != self.obs["ball_carrier"]:
                        """
                        If the agent is without ball,
                        <0 means no right movement action
                        """
                        if self.action_teamA[name] < 0:
                            reward += -1.0
                    else:
                        if flag == 0:
                            """
                            pass to empty locations
                            """
                            reward += -1.0
                        elif self.action_teamA[name] == -1:
                            """
                            this means trying to do invalid moving
                            """
                            reward += -1.0
                rewards.append(reward)
        return np.array(rewards), done

    def MoveToward(self, start, goal):

        move_list = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]

        d = np.array(goal) - np.array(start)
        if np.absolute(d[0]) + np.absolute(d[1]) > 1:
            if d[0] > 0 and d[1] < 0:
                if np.absolute(d[0]) > np.absolute(d[1]):
                    x, y = 1, 0
                else:
                    x, y = 0, -1
            elif d[0] < 0 and d[1] < 0:
                if np.absolute(d[0]) > np.absolute(d[1]):
                    x, y = -1, 0
                else:
                    x, y = 0, -1
            elif d[0] > 0 and d[1] > 0:
                if np.absolute(d[0]) > np.absolute(d[1]):
                    x, y = 1, 0
                else:
                    x, y = 0, 1
            elif d[0] < 0 and d[1] > 0:
                if np.absolute(d[0]) > np.absolute(d[1]):
                    x, y = -1, 0
                else:
                    x, y = 0, 1
            elif d[0] == 0:
                x, y = 0, d[1] / np.absolute(d[1])
            elif d[1] == 0:
                x, y = d[0] / np.absolute(d[0]), 0

        else:
            x = 0
            y = 0

        return move_list.index([x, y])
        # if [start[0] + x, start[1] + y] not in self.obs["teamA"]:
        #     if [start[0] + x, start[1] + y
        #         ] in self.obs["teamB"] and x + y != 0:
        #         return np.random.randint(5)

        #     return move_list.index([x, y])

        # else:
        #     """
        #         if the target location has been occupied,
        #         randomly choose an action
        #     """
        #     return np.random.randint(5)

    def OpponentPolicy(self):
        teamBpolicy = dict()

        blocked = False
        #if self.opponent_steps % 10 == 0:
        self.chasing_target = self.obs["ball_carrier"]
        self.opponent_steps = self.opponent_steps + 1
        if self.num_agents[1] == 0:
            """
                No opponents
            """
            teamBpolicy = {
                agent: np.random.randint(5)
                for agent in self.teamB_names
            }

        else:

            if self.extreme_mode is False:
                index = 0
                d_min = 1000
                for i in range(self.num_agents[1]):
                    d = np.linalg.norm(np.array(self.obs["teamA"][self.chasing_target])
                                       - np.array(self.obs["teamB"][i]))
                    if d < d_min:
                        index = i
                        d_min = d


                name = "agent%02d" % (index)
                move = self.MoveToward(
                    self.obs['teamB'][index],
                    self.obs['teamA'][self.chasing_target]
                )
                teamBpolicy[name] = move

                teamA_index = list(range(0, self.num_agents[0]))
                teamB_index = list(range(0, self.num_agents[1]))

                teamA_index.remove(self.chasing_target)
                teamB_index.remove(index)

                for i in range(self.num_agents[1]-1):

                    
                    name = "agent%02d" % (teamB_index[i])

                    # Create two lists deleting index, chasing target
                    # respectively.

                    # if i == index:
                    #     move = self.MoveToward(
                    #         self.obs["teamB"][i],
                    #         self.obs["teamA"][self.chasing_target])
                    if i < self.num_agents[0]-1:
                        # if i <= self.chasing_target:
                        #     move = self.MoveToward(self.obs["teamB"][i],
                        #                            self.obs["teamA"][i - 1])
                        # else:
                        move = self.MoveToward(self.obs["teamB"][teamB_index[i]],
                                               self.obs["teamA"][teamA_index[i]])
                    elif i >= self.num_agents[0]-1:
                        move = self.MoveToward(
                            self.obs["teamB"][teamB_index[i]],
                            self.obs["teamA"][self.chasing_target])
                    teamBpolicy[name] = move
            else:
                """
                    Extreme mode means all opponents chasing the agent with ball
                """
                for i in range(self.num_agents[1]):

                    name = "agent%02d" % (i)

                    move = self.MoveToward(
                        self.obs["teamB"][i],
                        self.obs["teamA"][self.chasing_target])
                    teamBpolicy[name] = move

        return teamBpolicy

    def CheckBlocked(self):
        for i in range(self.num_agents[1]):
            temp = np.array(self.obs["teamB"][i]) - np.array(
                self.obs["teamA"][self.obs["ball_carrier"]])
            if np.absolute(temp[0]) + np.absolute(temp[1]) <= 1:
                return True

        return False

    def MoveAction(self, action, location):
        move_list = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
        if self.mode == 1:
            action = action - self.length * self.width
            move = [math.floor(action / self.width), action % self.width]
        else:
            move = [math.floor(action / self.width), action % self.width]

        if ([move[0] - location[0], move[1] - location[1]] in move_list):
            return move_list.index(
                [move[0] - location[0], move[1] - location[1]])
        else:
            return -1

    def rtr_act(self, action):

        if self.use_tensor_action is True:
            return action
        else:
            new_action = np.zeros(action.shape)
            for i in range(self.num_agents[0]):
                if action[i] < self.num_agents[0]:
                    new_action[i] = self.obs["teamA"][action[i]][0] * self.width + self.obs["teamA"][action[i]][1]
                else:
                    move_list = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
                    new_action[i] = (self.obs["teamA"][i][0] + move_list[action[i]-self.num_agents[0]][0]) * self.width + self.obs["teamA"][i][1] + move_list[action[i]-self.num_agents[0]][1] + self.length * self.width
            return new_action
                    
    def step(self, action):
        """
            Step function

            If an agent's action hits another agent or hits the boundary,
            it would stay still.

            mode: 0, fixed movement actions
            mode: 1, two different action classes
            mode: 2, combine moving into passing, not used currently
        """

        action = self.rtr_act(action)
        
        self.steps = self.steps + 1
        ball_carrier = self.obs["ball_carrier"]

        self.action_teamA = {}
        passing_locations = self.length * self.width
        for i in range(self.num_agents[0]):
            name = "agent%02d" % (i)

            if self.mode == 0:
                self.action_teamA[name] = action[i] - passing_locations
                if self.action_teamA[name] >= 0:
                    self.action_teamA[name] = self.action_teamA[name] + 1
                if i == ball_carrier:
                    if action[i] < passing_locations:
                        passing = [
                            math.floor(action[i] / self.length),
                            action[i] % self.length
                        ]
                    else:
                        passing = self.obs["teamA"][i]

            elif self.mode == 2:
                self.action_teamA[name] = self.MoveAction(
                    action[i], self.obs["teamA"][i])

                if i == ball_carrier:
                    if self.action_teamA[name] < 0:
                        passing = [
                            math.floor(action[i] / self.length),
                            action[i] % self.length
                        ]
                    else:
                        passing = self.obs["teamA"][i]
            else:
                if action[i] >= self.length * self.width:
                    self.action_teamA[name] = self.MoveAction(
                        action[i], self.obs["teamA"][i])
                else:
                    self.action_teamA[name] = -2

                if i == ball_carrier:
                    if action[i] < self.length * self.width:
                        passing = [
                            math.floor(action[i] / self.length),
                            action[i] % self.length
                        ]
                    else:
                        passing = self.obs["teamA"][i]

        self.action_teamB = self.OpponentPolicy()
        new_action = dict()
        new_action["teamA"] = self.action_teamA
        new_action["teamB"] = self.action_teamB
        new_action["passing"] = passing

        # print(new_action)

        self.obs, debug_info = self.soccerfield.step(new_action)
        next_one = self.obs["next_one"]
        self.obs["ball_carrier"] = next_one

        flag = debug_info['flag']
        intercept = debug_info['intercept']
        
        blocked = self.CheckBlocked()
        reward, done = self.reward(flag, intercept, next_one, blocked)

        if self.steps >= self.max_steps:
            done = True
        self.draw()

        if self.use_state_feature is False:
            return self.Wrapper(self.obs['image'],self.obs['teamA']), reward, done, {}
        else:
            return self.rtr_state_feature(self.obs), reward, done, {}
        
    def rtr_state_feature(self, obs):
        rtr_obses = []

        for i in range(self.num_agents[0]):
            rtr_obs = []

            # Give information about if the agent is carrying the ball
            # or not
            rtr_obs = self.obs["teamA"][i] + [int(i==self.obs["ball_carrier"])]

            for k, loc in enumerate(self.obs["teamA"]):
                rtr_obs = rtr_obs + loc + [k]

            for k, loc in enumerate(self.obs["teamB"]):
                rtr_obs = rtr_obs + loc + [i + self.num_agents[0]]

            rtr_obses.append(rtr_obs)
        return np.array(rtr_obses).transpose()

        return np.array(rtr_obses)

    def Wrapper(self, img, state):

        if self.mode == 1:
            for i in range(self.num_agents[0]):
                img = np.concatenate((img, self.action_mask[i]), axis=-1)

            return img
            # state_img = []
            # for i in range(self.num_agents[0]):
            #     state_img.append(self.action_mask[i])
            # return state_img

        else:
            return img, state
        
    def draw(self):
        size = self.length * self.scale, self.width * self.scale
        ball_carrier = self.obs["ball_carrier"]
        if self.moving_penalty is True:
            self.step_mask = np.ndarray(
                shape=(size[0], size[1], 1), dtype=np.int32)
            self.step_mask[:, :, :] = self.moving_steps
        if self.mode == 0:
            action_mask = np.ndarray(
                shape=(size[0], size[1], 1), dtype=np.int32)
            action_mask[:, :, :] = 0

            x = self.obs["teamA"][ball_carrier][0] * self.scale
            y = self.obs["teamA"][ball_carrier][1] * self.scale
            action_mask[x:x + self.scale, :, 0] = 255
            action_mask[:, y:y + self.scale, 0] = 255

            for location in self.obs["teamA"]:
                x, y = location[0], location[1]
                x0, x1 = x - 1, x + 1
                y0, y1 = y - 1, y + 1
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                if x1 >= self.length:
                    x1 = self.length - 1
                if y1 >= self.width:
                    y1 = self.width - 1

                action_mask[x * self.scale:(x + 1) * self.scale,
                            y0 * self.scale:(y1 + 1) * self.scale, 0] = 255
                action_mask[x0 * self.scale:(x1 + 1) * self.scale,
                            y * self.scale:(y + 1) * self.scale, 0] = 255
            """
            for location in self.obs["teamA"]:
                x = self.obs["teamA"][ball_carrier][0] * self.scale
                y = self.obs["teamA"][ball_carrier][1] * self.scale
                action_mask[x:x+self.scale,:,0] = 255
                action_mask[:, y:y+self.scale, 0] = 255
            """

        else:
            self.action_mask = []

            x = self.obs["teamA"][ball_carrier][0] * self.scale
            y = self.obs["teamA"][ball_carrier][1] * self.scale

            # Combine two action masks into one

            # 50 for channel 1
            # 100 for channel 2
            # 150 means both actions are valid
            for location in self.obs["teamA"]:
                action_mask = np.ndarray(
                    shape=(size[0], size[1], 1), dtype=np.int32)
                action_mask[:, :, :] = 0
                x, y = location[0], location[1]

                x0, x1 = x - 1, x + 1
                y0, y1 = y - 1, y + 1
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                if x1 >= self.length:
                    x1 = self.length - 1
                if y1 >= self.width:
                    y1 = self.width - 1
                if x == self.obs["teamA"][ball_carrier][0] and y == self.obs["teamA"][ball_carrier][1]:
                    action_mask[x * self.scale:x * self.scale + self.scale, :,
                                0] = 50
                    action_mask[:, y * self.scale:y * self.scale + self.scale,
                                0] = 50

                action_mask[x * self.scale:(x + 1) * self.scale,
                            y0 * self.scale:(y1 + 1) * self.scale, 0] += 100
                action_mask[x0 * self.scale:(x1 + 1) * self.scale,
                            y * self.scale:(y + 1) * self.scale, 0] += 100
                if self.moving_penalty is False:
                    self.action_mask.append(action_mask)
                else:
                    self.action_mask.append(
                        np.concatenate((action_mask, self.step_mask), axis=2))
            
            # Action masks are two channel (old version)
            # for location in self.obs["teamA"]:
            #     action_mask = np.ndarray(
            #         shape=(size[0], size[1], 2), dtype=np.int32)
            #     action_mask[:, :, :] = 0
            #     x, y = location[0], location[1]

            #     x0, x1 = x - 1, x + 1
            #     y0, y1 = y - 1, y + 1
            #     if x0 < 0:
            #         x0 = 0
            #     if y0 < 0:
            #         y0 = 0
            #     if x1 >= self.length:
            #         x1 = self.length - 1
            #     if y1 >= self.width:
            #         y1 = self.width - 1
            #     if x == self.obs["teamA"][ball_carrier][0] and y == self.obs["teamA"][ball_carrier][1]:
            #         action_mask[x * self.scale:x * self.scale + self.scale, :,
            #                     0] = 255
            #         action_mask[:, y * self.scale:y * self.scale + self.scale,
            #                     0] = 255

            #     action_mask[x * self.scale:(x + 1) * self.scale,
            #                 y0 * self.scale:(y1 + 1) * self.scale, 1] = 255
            #     action_mask[x0 * self.scale:(x1 + 1) * self.scale,
            #                 y * self.scale:(y + 1) * self.scale, 1] = 255
            #     if self.moving_penalty is False:
            #         self.action_mask.append(action_mask)
            #     else:
            #         self.action_mask.append(
            #             np.concatenate((action_mask, self.step_mask), axis=2))
        try:
            screen = pygame.display.set_mode(size, 0, 32)

            pygame.display.set_caption("Take-the-Treasure")

        except:
            os.environ["SDL_VIDEODRIVER"] = 'dummy'
            screen = pygame.display.set_mode(size, 0, 32)
            pygame.display.set_caption("Take-the-Treasure")

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
        self.obs["image"]= np.array(img)[..., np.newaxis]
        

        if self.masking is True and self.mode == 0:            
            self.obs["image"] = np.concatenate(
                (self.obs["image"], action_mask), axis=2)

        return

    def render(self):
        """
            rendering of the environment
            blue for team A
            yellow for team B
        """
        self.draw()
        """
            In case the display mode does not work
        """
        try:
            pygame.display.update()

        except:
            return

    def record(self, save_dir, save_filename):
        scale = 20
        size = self.length * scale, self.width * scale
        ball_carrier = self.obs["ball_carrier"]
        screen = pygame.display.set_mode(size)

        pygame.display.set_caption("GridSoccer")
        screen.fill(green)

        for location in self.obs["teamA"]:
            pygame.draw.rect(screen, blue, [
                location[0] * scale, location[1] * scale, scale, scale
            ])

        for location in self.obs["teamB"]:

            pygame.draw.rect(screen, yellow, [
                location[0] * scale, location[1] * scale, scale, scale
            ])

        pygame.draw.rect(screen, red, [
            self.obs["teamA"][ball_carrier][0] * scale,
            self.obs["teamA"][ball_carrier][1] * scale, scale, scale
        ])
        save_path = os.path.join(save_dir, save_filename)

        pygame.display.update()
        pygame.image.save(screen, save_path)

    def rtr_state(self, index):
        screen = pygame.display.set_mode((self.length * self.scale,
                                          self.width * self.scale))
        pygame.display.set_caption("GridSoccer")
        screen.fill(green)
        location = self.obs["teamA"][index]
        if index == self.obs["ball_carrier"]:
            color = red
        else:
            color = blue

        pygame.draw.rect(screen, color, [
            location[0] * self.scale, location[1] * self.scale, self.scale,
            self.scale
        ])
        return pygame.surfarray.array3d(screen)

    def rtr_red(self):
        screen = pygame.display.set_mode((self.length * self.scale,
                                          self.width * self.scale))
        screen.fill(red)
        return pygame.surfarray.array3d(screen)

    def rtr_blue(self):
        screen = pygame.display.set_mode((self.length * self.scale,
                                          self.width * self.scale))
        screen.fill(blue)
        return pygame.surfarray.array3d(screen)

    def rtr_yellow(self):
        screen = pygame.display.set_mode((self.length * self.scale,
                                          self.width * self.scale))
        screen.fill(yellow)
        return pygame.surfarray.array3d(screen)

    def rtr_green(self):
        screen = pygame.display.set_mode((self.length * self.scale,
                                          self.width * self.scale))
        screen.fill(green)
        return pygame.surfarray.array3d(screen)



    # def set(self,
    #         length=10,
    #         width=10,
    #         num_agents=(2, 2),
    #         max_steps=200,
    #         scale=20,
    #         masking=False,
    #         mode=0,
    #         extreme_mode=False,
    #         moving_penalty=False,
    #         chip_kick=False):
    #     self.length = length
    #     self.width = width
    #     self.scale = scale
    #     self.num_agents = num_agents
    #     self.soccerfield.length = length
    #     self.soccerfield.width = width
    #     self.soccerfield.scale = scale
    #     self.soccerfield.num_agents = num_agents
    #     self.teamA_names = ["agent%02d" % (i) for i in range(num_agents[0])]
    #     self.teamB_names = ["agent%02d" % (i) for i in range(num_agents[1])]
    #     self.max_steps = max_steps

    #     self.masking = masking
    #     self.mode = mode  # mode 0: passing and moving separated
    #     self.extreme_mode = extreme_mode

    #     self.moving_penalty = moving_penalty
    #     self.moving_steps = 0

    #     self.action_teamA = Dict(
    #         {agent: Discrete(5)
    #          for agent in self.teamA_names})

    #     self.action_teamB = Dict(
    #         {agent: Discrete(5)
    #          for agent in self.teamB_names})

    #     self.action_space = Dict({
    #         "passing":
    #         Discrete(self.length * self.width),
    #         "move":
    #         self.action_teamA
    #     })
    #     self.observation_space = Box(self.length * self.scale,
    #                                  self.width * self.scale, 3)
    
