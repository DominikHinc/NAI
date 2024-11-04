import math
import random

MAX_STEER = math.pi / 10
MAX_SPEED = 1
MAX_REV_SPEED = -3
MAX_DIST = 700

EMPTY_MAP = (
    '################',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '#              #',
    '################',
)
MAP1 = (
    '#################',
    '##             ##',
    '#     ###       #',
    '#           ##  #',
    '####     #      #',
    '#               #',
    '#   ##       #  #',
    '#               #',
    '# #             #',
    '#       ####    #',
    '#################',
)
MAP2 = (
    '################',
    '##            ##',
    '#    # ## #    #',
    '#    # ## #    #',
    '#              #',
    '#              #',
    '#   ########   #',
    '#              #',
    '#              #',
    '#        ##    #',
    '##            ##',
    '################',
)
MAP = MAP1
#ustawiamy sobię mapę jaką chcemy albo rysujemy nową
TILE_SIZE = 50
MAP_SIZE = (len(MAP[0]), len(MAP))
SCREEN_HEIGHT = MAP_SIZE[1] * TILE_SIZE
SCREEN_WIDTH = MAP_SIZE[0] * TILE_SIZE
MAX_DEPTH = int(max(*MAP_SIZE) * TILE_SIZE)

'''
    Randomize position

    :param pos: the position to randomize
    :param diff: the difference to randomize

    :return: the randomized position
'''
def randomize_pos(pos, diff):
    return [pos[0] + random.randint(-diff, diff), pos[1] + random.randint(-diff, diff)]