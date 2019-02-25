"""Implementation of soccer agentr"""
# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import attr
import numpy as np

from enum import Enum

class PassStatus(Enum):
    HOLDBALL = 0
    PASS_IN_ROW_OR_COLUMN = 1
    PASS_NOT_IN_ROW_OR_COLUMN = 2
    PASS_NOT_TO_TEAMMATE = 3

@attr.s
class SoccerAction():

    move_x = attr.ib(default=0)
    move_y = attr.ib(default=0)
    passing = attr.ib(default=(-1, -1))


@attr.s
class SoccerAgent():

    loc_x = attr.ib(default=0)
    loc_y = attr.ib(default=0)

    action = attr.ib(default=SoccerAction())
    with_ball = attr.ib(default=False)

    move_x = attr.ib(default=[0, 1, 0, -1, 0])
    move_y = attr.ib(default=[0, 0, 1, 0, -1])

    def step(self, length, width, agent_locations):
        """
            This part is to move the agent while checking
            if it hits the boundary or other agents
        """
        tempx = self.loc_x + self.action.move_x
        tempy = self.loc_y + self.action.move_y

        if [tempx, tempy] not in agent_locations:
            if tempx < 0 or tempx >= length:
                self.loc_x = self.loc_x
            else:
                self.loc_x = self.loc_x + self.action.move_x

            if tempy < 0 or tempy >= width:
                self.loc_y = self.loc_y
            else:
                self.loc_y = self.loc_y + self.action.move_y

        return self.loc_x, self.loc_y

    def move(self, index):
        if index < 0:
            index = 0
        return self.move_x[index], self.move_y[index]


@attr.s
class SoccerField():

    length = attr.ib(default=10)
    width = attr.ib(default=10)
    scale = attr.ib(default=20)
    num_agents = attr.ib(default=(2, 2))
    teamA = attr.ib(default=attr.Factory(list))
    teamB = attr.ib(default=attr.Factory(list))
    next_one = attr.ib(default=0)
    """Adding agents"""

    def reset(self):
        """
            Reset the environment, including agents'
            actions and locations

            Specify a target agent, could be unused

            Specify an agent with the ball
        """
        for i in range(self.num_agents[0]):
            action = SoccerAction()
            self.teamA.append(SoccerAgent(0, 0, action))

        for i in range(self.num_agents[1]):
            action = SoccerAction()
            self.teamB.append(SoccerAgent(0, 0, action))
        self.agent_locations = []
        self.ball = np.random.randint(self.num_agents[0])

        self.next_one = self.ball

        self.teamA[self.ball].action.with_ball = True

        count = 0

        while count < self.num_agents[0]:
            """"
            If it is the ball carrier, passing is True.
            Otherwise, passing is False
            """
            x = np.random.randint(self.length)
            y = np.random.randint(self.width)

            if [x, y] not in self.agent_locations:
                self.agent_locations.append([x, y])
                self.teamA[count].loc_x = x
                self.teamA[count].loc_y = y
                count = count + 1

        count = 0
        while count < self.num_agents[1]:
            x = np.random.randint(self.length)
            y = np.random.randint(self.width)

            if [x, y] not in self.agent_locations:
                self.agent_locations.append([x, y])
                self.teamB[count].loc_x = x
                self.teamB[count].loc_y = y
                count = count + 1

        self.obs = {
            "teamA": self.agent_locations[0:self.num_agents[0]],
            "teamB": self.agent_locations[self.num_agents[0]:],
            "ball_carrier": self.ball,
            "next_one": self.next_one
        }
        return self.obs

    def step(self, action):
        """
            Every agent moves a step
        """
        debug_info = dict()
        flag = PassStatus.HOLDBALL
        
        intercept = 0

        valid_actions = True
        
        self.ball = self.next_one
        self.teamA[self.ball].action.passing = action["passing"]
        ball_carrier = self.teamA[self.ball]
        passing_pos = ball_carrier.action.passing

        if passing_pos != [ball_carrier.loc_x, ball_carrier.loc_y]:
            x, y = passing_pos[0] - ball_carrier.loc_x, passing_pos[1] - ball_carrier.loc_y
            if (x != 0 and y != 0):
                valid_actions = False
            if passing_pos in self.obs["teamA"]:
                if x == 0 or y == 0:
                    flag = PassStatus.PASS_IN_ROW_OR_COLUMN
                    self.next_one = self.obs["teamA"].index(passing_pos)

                    for location in self.agent_locations:
                        if location != [
                                ball_carrier.loc_x, ball_carrier.loc_y
                        ]:
                            if (
                                    location[0] - passing_pos[0] == 0
                                    and np.absolute(location[1] - ball_carrier.
                                                    loc_y) < np.absolute(y) and (location[1] - ball_carrier.loc_y) * (location[1] - passing_pos[1]) < 0
                            ) or (location[1] - passing_pos[1] == 0 and
                                  np.absolute(location[0] - ball_carrier.loc_x)
                                  < np.absolute(x) and  (location[0] - ball_carrier.loc_x) * (location[0] - passing_pos[0]) < 0):
                                intercept = 1
                else:
                    flag = PassStatus.PASS_NOT_IN_ROW_OR_COLUMN
            else:
                flag = PassStatus.PASS_NOT_TO_TEAMMATE

        agent_locations = []
        for i in range(self.num_agents[0]):
            name = "agent%02d" % (i)
            if i == self.ball:
                move_x, move_y = self.teamA[i].move(action["teamA"][name])
                passing = action["passing"]

            else:
                move_x, move_y = self.teamA[i].move(action["teamA"][name])
                passing = (-1, -1)

            self.teamA[i].action.move_x = move_x
            self.teamA[i].action.move_y = move_y
            self.teamA[i].action.passing = passing

            self.teamA[i].step(self.length, self.width, self.agent_locations)
            self.agent_locations[i] = [
                self.teamA[i].loc_x, self.teamA[i].loc_y
            ]

        for i in range(self.num_agents[1]):
            name = "agent%02d" % (i)
            move_x, move_y = self.teamB[i].move(action["teamB"][name])
            passing = 1

            self.teamB[i].action.move_x = move_x
            self.teamB[i].action.move_y = move_y
            self.teamB[i].action.passing = passing

            self.teamB[i].step(self.length, self.width, self.agent_locations)
            self.agent_locations[self.num_agents[0] + i] = [
                self.teamB[i].loc_x, self.teamB[i].loc_y
            ]

        for i in range(self.num_agents[0]):
            agent_locations.append([self.teamA[i].loc_x, self.teamA[i].loc_y])
        for i in range(self.num_agents[1]):
            agent_locations.append([self.teamB[i].loc_x, self.teamB[i].loc_y])

        self.agent_locations = agent_locations

        self.obs = {
            "teamA": self.agent_locations[0:self.num_agents[0]],
            "teamB": self.agent_locations[self.num_agents[0]:],
            "ball_carrier": self.ball,
            "next_one": self.next_one
        }
        debug_info["flag"] = flag
        debug_info["intercept"] = intercept
        debug_info["in_row_column"] = valid_actions
        return self.obs, debug_info
