"""Engine for breakout"""
# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import pygame
import numpy as np
import os

from collections import deque
from PIL import Image
import math

SCREEN_SIZE = 84, 84

BRICK_WIDTH   = 12
BRICK_HEIGHT  = 3
PADDLE_WIDTH  = 12
PADDLE_HALF_WIDTH = 6
PADDLE_HEIGHT = 5
BALL_DIAMETER = 4
BALL_RADIUS   = 2
MAX_BALL_VEL = 2

MAX_BALL_X   = SCREEN_SIZE[0] - BALL_DIAMETER
MAX_BALL_Y   = SCREEN_SIZE[1] - BALL_DIAMETER

BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE  = (0,0,255)
WHITE = (255,255,255)

BRICK_COLOR = [(200,0,0),(200,200,0), (0,200,0), (0,0,200)]

STATE_BALL_IN_PADDLE = 0
STATE_PLAYING = 1
STATE_WON = 2
STATE_GAME_OVER = 3

WIDTH, HEIGHT = 84, 84
border = [[1,0],[0,1],[-1,0],[0,-1]]

class Brick(object):
    def __init__(self, rect, hit=0):
        self._rect = rect
        # 0 - not hit 1 - hit
        self._hit = hit

class PyGameEngine(object):
    PPM = 10.0
    TARGET_FPS = 60
    TIME_STEP = 1 / TARGET_FPS
    def __init__(self, caption, world):
        self.width = world["width"]
        self.height = world["height"]
        self.rows = int(6 * self.height / HEIGHT) 
        self.cols = int(self.width / BRICK_WIDTH)
        self.size = (self.width, self.height)
        self.caption = caption
        self.set_limit()
        self.valid = True
        self.state_break = False
        self.use_masking = True

        self.num_bricks = 0

        self.bricks = []
        
        self.history = world["history"]

        self.clock = pygame.time.Clock()
        self.init_game()

    def init_game(self):
        try:
            self.screen = pygame.display.set_mode(self.size, 0, 32)
            pygame.display.set_caption(self.caption)
        except:
            os.environ["SDL_VIDEODRIVER"] = 'dummy'
            self.screen = pygame.display.set_mode(self.size, 0, 32)
            pygame.display.set_caption(self.caption)
        self.state = STATE_PLAYING

        self.lives = 5

        self.set_objects()
        self.create_env()
        self.update_render()

    def set_objects(self):
        """Set paddle"""
        # "Try fix paddle position for every episode"
        # if self.lives == 5:
        #     paddle_pos = self.width / 2 + 7
        #     self.paddle = pygame.Rect(
        #         paddle_pos,
        #         self.height - PADDLE_HEIGHT - 3,
        #         PADDLE_WIDTH,
        #         PADDLE_HEIGHT
        #     )

        paddle_pos = np.random.randint(self.width-PADDLE_WIDTH)
        self.paddle = pygame.Rect(
            paddle_pos,
            self.height - PADDLE_HEIGHT - 3,
            PADDLE_WIDTH,
            PADDLE_HEIGHT
        )
        """Set ball"""
        ball_pos = np.random.randint(self.width)

        if ball_pos < int(self.width/2):
            self.ball_vel = [MAX_BALL_VEL, MAX_BALL_VEL]
        else:
            self.ball_vel = [-MAX_BALL_VEL, MAX_BALL_VEL]

        self.ball = pygame.Rect(
            ball_pos-BALL_RADIUS,
            int(self.height / 3 * 2)-BALL_RADIUS,
            BALL_DIAMETER,
            BALL_DIAMETER)
        self.obs_que = deque()
        

    def create_env(self):
        y_ofs = int(20 * self.height / HEIGHT) + (self.rows-1)*BRICK_HEIGHT

        if self.bricks == []:
            for i in range(self.rows):
                x_ofs = 0
                for j in range(self.cols):
                    self.bricks.append(Brick(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT)))
                    x_ofs += BRICK_WIDTH
                y_ofs -= BRICK_HEIGHT

        else:
            for brick in self.bricks:
                brick._hit = 0
        self.num_bricks = len(self.bricks)

    def draw_env(self):
        for brick in self.bricks:
            if brick._hit == 0:
                index = int((brick._rect.top - int(20*self.width / WIDTH)) / BRICK_HEIGHT)
                pygame.draw.rect(self.screen, BRICK_COLOR[index % 4], brick._rect)
            
    def update(self, world):
        self.width = world["width"]
        self.height = world["height"]
        self.history = world["history"]
        self.use_masking = world["masking"]

        self.size = (self.width, self.height)
        self.set_limit()

        self.rows = int(6 * self.height / HEIGHT)
        self.cols = int(7 * self.width / WIDTH)

    def set_limit(self):
        self.max_horizontal_move = 5

    def get_obs(self, use_state_feature=False):
        if use_state_feature is False:
            screen_array = pygame.surfarray.array3d(self.screen)
            self.img = Image.fromarray(screen_array)
            self.img = self.img.convert('L')
            self.img = np.array(self.img)[...,np.newaxis]
            if self.use_masking is True:
                self.img = np.concatenate(
                    (np.array(self.img, dtype='uint8'), self.action_masking()),
                    axis=2)

                # img = self.action_masking(np.array(img, dtype='uint8'))
            # self.obs = np.concatenate((self.obs, img), axis=2)
            self.obs = self.img
        else:
            self.obs = []
            self.obs = self.obs + [self.ball.left, self.ball.top]
            self.obs = self.obs + [self.paddle.left, self.paddle.top]

            self.obs = self.obs + [int(20 * self.height / HEIGHT) + (self.rows-1)*BRICK_HEIGHT]
            
            for brick in self.bricks:
                self.obs = self.obs + [brick._hit]

        return self.obs

    def action_masking(self):
        action_mask = np.ndarray(shape=(self.width, self.height, 1), dtype = np.int32)
        action_mask[:, :, :] = 0
        x = self.paddle.left

        if x - self.max_horizontal_move<0:
            action_left_limit = 0
        else:
            action_left_limit = x - self.max_horizontal_move

        if x + self.max_horizontal_move > self.width - PADDLE_WIDTH:
            action_right_limit = self.width - PADDLE_WIDTH
        else:
            action_right_limit = x + self.max_horizontal_move

        action_mask[
            action_left_limit+PADDLE_HALF_WIDTH:action_right_limit+PADDLE_HALF_WIDTH,
            self.paddle.top,
            0] = 255
        return action_mask
        
    def render(self):
        pygame.display.flip()

    def record(self, save_path):
        pygame.display.flip()
        pygame.image.save(self.screen, save_path)

    def get_img(self):
        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()

    def update_render(self):

        self.screen.fill(BLACK)
        self.draw_env()
        pygame.draw.rect(
            self.screen,
            BLUE,
            self.paddle
        )
        pygame.draw.circle(
            self.screen,
            WHITE,
            (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS),
            BALL_RADIUS
        )
        
    def step(self,action):
        """move_ball, handle_collisions, check_input come in here"""
        #self.clock.tick(50)
        self.move_paddle(action)
        if self.state == STATE_PLAYING:
            self.move_ball()
            self.handle_collisions()

        self.update_render()

    def reward(self):
        reward = 0

        if self.state_break is True:
            self.state_break = False
            reward = 1
        elif self.state == STATE_WON or self.state == STATE_GAME_OVER:
            reward = 0
        elif self.valid is False:
            reward += -1
        return reward

    def valid_action(self, action):
        self.valid = True
        action[0] -= PADDLE_HALF_WIDTH
        if action[1]!=self.paddle.top:
            # print('not equal to top')
            self.valid = False
            return 0
        elif action[0]-self.paddle.left < -self.max_horizontal_move: #or action[0]<0:
            # print('out of left bound')
            self.valid = False
            return 0
        elif action[0] - self.paddle.left > self.max_horizontal_move: # or action[0]>self.width - PADDLE_WIDTH:
            # print('out of right bound')
            self.valid = False
            return 0
        else:
            return action[0] - self.paddle.left

    def move_paddle(self, action):
        self.paddle.left += action

        if self.paddle.left < 0:
            self.paddle.left = 0
        if self.paddle.left > self.width - PADDLE_WIDTH:
            self.paddle.left = self.width - PADDLE_WIDTH

    def get_state(self):
        if self.state == STATE_WON or self.state == STATE_GAME_OVER:
            return True
        else:
            return False

    def move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top += self.ball_vel[1]
        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= self.width - BALL_DIAMETER:
            self.ball.left = self.width - BALL_DIAMETER
            self.ball_vel[0] = -self.ball_vel[0]

        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= self.height - BALL_DIAMETER:
            self.ball.top = self.height - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]

    def determine_collide_side(self, brick, point):
        border = [[1,0],[0,1],[-1,0],[0,-1]]
        for i in range(4):
            point1 =(point[0]+border[i][0], point[1]+border[i][1])
            if brick.left<=point1[0] and brick.right>=point1[0] and brick.top<=point1[1] and brick.bottom>=point1[1]:
                # print("Brick", brick.left, brick.top, brick.right, brick.bottom)
                if i==0:
                    return "left"
                elif i==1:
                    return "top"
                elif i==2:
                    return "right"
                else:
                    return "bottom"
    def check_collide(self, brick, point):
        for i in range(4):
            point1 =(point[0]+border[i][0], point[1]+border[i][1])
            if brick.left<=point1[0] and brick.right>=point1[0] and brick.top<=point1[1] and brick.bottom>=point1[1]:
                return True
        return False

    def handle_collisions(self):
        old_ball = (self.ball.centerx - self.ball_vel[0],self.ball.centery - self.ball_vel[1])
        if self.ball_vel[0]>0:
            sign_x = 1
        else:
            sign_x = -1

        if self.ball_vel[1]>0:
            sign_y = 1
        else:
            sign_y = -1

        temp_vely = self.ball_vel[1]
        temp_velx = self.ball_vel[0]

        for i in range(MAX_BALL_VEL):
            # print(self.ball.left, self.ball.top)
            # print(old_ball)
            x = old_ball[0] + sign_x * (i+1)
            y = old_ball[1] + sign_y * (i+1)
            for brick in self.bricks:
                if brick._hit == 0 and self.check_collide(brick._rect, (x,y)) is True:
                   relative_pos = self.determine_collide_side(brick._rect, (x,y))
                   # print(relative_pos)
                   # input()
                   if relative_pos == "top" or relative_pos == "bottom":
                       temp_vely = -self.ball_vel[1]
                   else:
                       temp_velx = -self.ball_vel[0]
                   self.ball.left = x-1
                   self.ball.top = y-1
                   # self.bricks.remove(brick)
                   brick._hit = 1
                   self.state_break = True 
            if self.state_break is True:
                break
        self.ball_vel[1] = temp_vely
        self.ball_vel[0] = temp_velx
        if len(self.bricks) == 0:
            self.state = STATE_WON

        if self.ball.colliderect(self.paddle):
            self.ball.top = self.paddle.top - BALL_DIAMETER
            self.ball_vel[1] = - self.ball_vel[1]
            if self.ball.left + BALL_RADIUS < self.paddle.left + int(PADDLE_WIDTH / 2):
                self.ball_vel[0] = -MAX_BALL_VEL
            else:
                self.ball_vel[0] = MAX_BALL_VEL
        elif self.ball.top+BALL_RADIUS > self.paddle.top:
            self.lives -= 1
            if self.lives <= 0:
                self.state = STATE_GAME_OVER
            else:
                self.set_objects()

    def paddle_left(self):
        return [self.paddle.left, self.paddle.top]

    def paddle_middle(self):
        return [self.paddle.left+PADDLE_HALF_WIDTH, self.paddle.top]
