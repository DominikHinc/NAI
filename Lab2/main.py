

'''
Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
The problem is to implement a vacuum cleaner agent that can navigate through the environment and clean the dirt. 
The agent should be able to sense the environment and make decisions based on the information it gathers. 
The agent should be able to move around the environment, avoid obstacles, and clean the dirt efficiently.


Solution:
Apply fuzzy logic to control the vacuum cleaner agent.

System architecture:

front  -|   Fuzzy Controller   |- acc
        |                      |
vel    -|                      |- ang_acc
        |                      |
ang_vel-|                      |

The fuzzy controller takes three inputs:
- front: the distance to the nearest obstacle in front of the vacuum cleaner
- vel: the current velocity of the vacuum cleaner
- ang_vel: the current angular velocity of the vacuum cleaner

The fuzzy controller has two outputs:
- acc: the acceleration of the vacuum cleaner
- ang_acc: the angular acceleration of the vacuum cleaner

The fuzzy controller has the following rules:
- if the distance to the nearest obstacle in front of the vacuum cleaner is far, the vacuum cleaner should accelerate
- if the distance to the nearest obstacle in front of the vacuum cleaner is medium, the vacuum cleaner should turn left or right depending on the angular velocity (default is left)
- if the distance to the nearest obstacle in front of the vacuum cleaner is close, the vacuum cleaner should turn sharply left or right depending on the angular velocity (default is left)


Setup:

IMPORTANT
Make sure you are using python 3.9.6, newer versions cause library compatibility issues.

Install dependencies:

pip install -r requirements.txt


Run the game:

python3 ./main.py


You can find preview of how this works in: PREVIEW.mov

Sources:
https://journals.sagepub.com/doi/full/10.5772/54427
https://stackoverflow.com/questions/50283552/creating-a-circle-in-pygame-with-points-on-circumference
https://github.com/a-float/psi-fuzzy-driving

'''

import pygame
import sys
import math
from vacuum import Vacuum
from maps import *
from random import random

CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
vacuum = Vacuum(randomize_pos(CENTER, 40), random() * 2 * math.pi, (255, 0, 0))

pygame.init()

win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption('Rumba')
clock = pygame.time.Clock()

'''
    Draw the map
'''
def draw_map():
    for row in range(MAP_SIZE[1]):
        for col in range(MAP_SIZE[0]):
            pygame.draw.rect(
                win,
                (98, 81, 64) if MAP[row][col] == '#' else (173, 151, 129),
                (col * TILE_SIZE + 1, row * TILE_SIZE + 1, TILE_SIZE - 2, TILE_SIZE - 2)
            )


'''
    Main simulation loop
'''
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    pygame.draw.rect(win, (173, 151, 129), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
    draw_map()

    # apply raycasting
    dists = vacuum.cast_ray(win)
    vacuum.update(dists)
    vacuum.move()

    pygame.display.flip()
    clock.tick(30)