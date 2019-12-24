import pygame
from pygame.locals import QUIT
# from pygame.locals import KEYDOWN, K_LEFT, K_RIGHT

import numpy as np
import pandas as pd

from neural import NeuralNetwork

SCREEN_SIZE = (600, 500)


def collision(bar_pos, bar_size, ball_pos):
    x = True if(ball_pos[0] >= bar_pos[0] and
                ball_pos[0] <= bar_pos[0]+bar_size) else False
    return (x and bar_pos[1] == ball_pos[1])


class Bar:
    def __init__(self, size, position, velocity, collor):
        self.size = size
        self.position = position
        self.velocity = velocity
        self.collor = collor

        self.draw_bar = pygame.Surface(size)
        self.draw_bar.fill(collor)

    def move(self, direction):

        if direction:
            soma = self.position[0]+self.size[0]+self.velocity
            if soma <= SCREEN_SIZE[0]:
                x = self.position[0] + 10
                y = self.position[1]
                self.position = (x, y)

        else:
            soma = self.position[0]-self.velocity
            if soma >= 0:
                x = self.position[0] - 10
                y = self.position[1]
                self.position = (x, y)


class Ball:
    def __init__(self, size, velocity, collor):
        self.size = size
        self.velocity = velocity
        self.collor = collor

        self.draw_ball = pygame.Surface(size)
        self.draw_ball.fill(collor)

        self.position = None
        self.new_position_on_grid_random()

    def new_position_on_grid_random(self):
        self.position = (np.random.randint(0, 390)//10 * 10, 0)

    def move(self):
        new_y = self.position[1]+5

        if new_y > SCREEN_SIZE[1]:
            self.new_position_on_grid_random()
            return 1

        else:
            self.position = (self.position[0], new_y)
            return 0


pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Ping')

font = pygame.font.Font(None, 36)

bar = Bar((100, 10),
          (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]-10),
          10,
          (255, 255, 255))

ball = Ball((10, 10), 5, (255, 255, 255))

clock = pygame.time.Clock()

direction = 0
score = 0

network = NeuralNetwork(20, 0.1)
generation = 0
error_count = 0

df = pd.read_csv('data.csv')

data = []

for index, row in df.iterrows():
    data.append([row['ball'], row['bar']])

data = np.array(data)

resp = np.array(df['direction'])

'''
df = pd.DataFrame(columns=['ball', 'bar', 'direction'])
index = 0
'''

while True:

    clock.tick(20)

    for event in pygame.event.get():
        if event.type == QUIT:
            # df.to_csv('data2.csv', index=False)
            pygame.quit()

        '''
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                direction = 0

            elif event.key == K_RIGHT:
                direction = 1
        '''

    if collision(bar.position, bar.size[0], ball.position):
        ball.new_position_on_grid_random()
        score += 1

    else:
        error = ball.move()
        error_count += error

    if error_count == 1:
        network.train(data, resp)
        generation += 1
        score = 0
        error_count = 0

    ball_pos = ball.position[0] - SCREEN_SIZE[0]/2
    bar_pos = bar.position[0] - SCREEN_SIZE[0]/2
    bar_pos += bar.size[0]/2

    ball_pos = ball_pos/100
    bar_pos = bar_pos/100

    direction = network.feedforward(np.array([ball_pos, bar_pos]))
    direction = 1 if direction > 0.5 else 0
    '''
    df.loc[index] = [ball_pos, bar_pos, direction]
    index += 1
    '''

    bar.move(direction)

    text = font.render('Score: '+str(score), 1, (255, 255, 255))
    text2 = font.render('Geração: '+str(generation), 1, (255, 255, 255))

    screen.fill((0, 0, 0))

    screen.blit(text, (10, 10))
    screen.blit(text2, (10, 40))

    screen.blit(ball.draw_ball, ball.position)

    screen.blit(bar.draw_bar, bar.position)

    pygame.display.update()
