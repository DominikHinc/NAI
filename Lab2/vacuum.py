import numpy as np
import math
import pygame
from fuzzy_controller import controller
from maps import *

VACUUM_SIZE = 50

class Vacuum:

    '''
        Initialize the vacuum cleaner

        :param start_pos: the starting position of the vacuum cleaner
        :param start_angle: the starting angle of the vacuum cleaner
        :param color: the color of the vacuum cleaner
    '''
    def __init__(self, start_pos, start_angle, color):
        self.acc = 0
        self.color = color
        self._speed = 0
        self._pos = list(start_pos)
        self._angle = start_angle
        self._ang_vel = 0
        self.ang_acc = 0
        self.FOV = math.pi / 1.9
        self.image = pygame.transform.scale(pygame.image.load("rumba.png"), (VACUUM_SIZE, VACUUM_SIZE))


    '''
        Update the vacuum cleaner

        :param distances: the distances to the nearest obstacles
    '''
    def update(self, distances):
        dists = np.array(distances) ** (1 / 2)
        controller.input['front'] = dists[0]
        controller.input['vel'] = self._speed
        controller.input['ang_vel'] = np.rad2deg(self._ang_vel)

        try:
            controller.compute()
            self.acc = controller.output['acc']
            self.ang_acc = np.deg2rad(controller.output['ang_acc']) * 3
        except ValueError as e:
            print("Error:")
            print(str(e))
            print(f"on input:\n{controller.input}")


        self.move()

    '''
        Respawn the vacuum cleaner
    '''
    def respawn(self, pos):
        self._pos = np.array(pos)

    '''
        Move the vacuum cleaner
    '''
    def move(self):
        # update angle
        self._ang_vel = Vacuum.clamp(self._ang_vel + self.ang_acc, -MAX_STEER, MAX_STEER)
        self._angle += self._ang_vel * (7 / (1.5 + abs(self._speed))) * self._speed / MAX_SPEED
        self.ang_acc = 0
        # update velocity
        self._speed = Vacuum.clamp(self._speed + self.acc, MAX_REV_SPEED, MAX_SPEED)
        self.acc = 0
        # apply velocity with turning
        self._pos += Vacuum.rotate_point(self._speed, self._angle)
        self._speed *= 0.98
        self._ang_vel *= 0.9

    ''' 
        Cast ray from the vacuum cleaner
        
        :param win: the window to draw the ray

        :return: the distances to the nearest obstacles
    '''
    def cast_ray(self, win):
        # define left most angle of FOV
        start_angle = (self.angle)
        distances = []
        # loop over casted rays

        for depth in range(MAX_DEPTH):
            # get ray target coordinates
            target_x = self.x - math.sin(start_angle) * depth
            target_y = self.y + math.cos(start_angle) * depth

            # covert target X, Y coordinate to map col, row
            col = int(target_x / TILE_SIZE)
            row = int(target_y / TILE_SIZE)

            # ray hits the condition
            if MAP[row][col] == '#':
                # highlight wall that has been hit by a casted ray
                pygame.draw.rect(win, (98, 81, 64), (col * TILE_SIZE + 1,
                                                    row * TILE_SIZE + 1,
                                                    TILE_SIZE - 2,
                                                    TILE_SIZE - 2))

                # draw casted ray
                pygame.draw.line(win, (0,0,0), (self.x, self.y), (target_x, target_y))

                # get the precise distance to the wall
                p1 = ((self.x, self.y), (target_x, target_y))  # ray
                if target_y < self.y:
                    p2 = ((0, (row + 1) * TILE_SIZE), (1, (row + 1) * TILE_SIZE))  # bottom horizontal line
                else:
                    p2 = ((0, row * TILE_SIZE), (1, row * TILE_SIZE))  # top horizontal line
                if target_x > self.x:
                    p3 = ((col * TILE_SIZE, 0), (col * TILE_SIZE, 1))  # left vertical line
                else:
                    p3 = (((col + 1) * TILE_SIZE, 0), ((col + 1) * TILE_SIZE, 1))  # right vertical line
                int1 = line_intersection(p1, p2)
                int2 = line_intersection(p1, p3)
                d1 = sq_dist((self.x, self.y), int1)
                d2 = sq_dist((self.x, self.y), int2)
                if (int1[0] < min(self.x, target_x)) or (int1[1] < min(self.y, target_y)) or (
                        int1[0] > max(self.x, target_x) or (int1[1] > max(self.y, target_y))):
                    distances.append(d2)
                elif (int2[0] < min(self.x, target_x)) or (int2[1] < min(self.y, target_y)) or (
                        int2[0] > max(self.x, target_x) or (int2[1] > max(self.y, target_y))):
                    distances.append(d1)
                else:
                    distances.append(max(d1, d2))
                break

        rotated_image = pygame.transform.rotate(self.image, -math.degrees(self._angle))
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x - VACUUM_SIZE / 2, self.y - VACUUM_SIZE / 2)).center)
        win.blit(rotated_image, new_rect.topleft)

        return distances

    '''
        Clamp a value between a minimum and a maximum value

        :param val: the value to clamp
        :param min_val: the minimum value
        :param max_val: the maximum value

        :return: the clamped value
    '''
    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min_val, min(val, max_val))

    '''
        Rotate a point

        :param speed: the speed of the rotation
        :param angle: the angle of the rotation

        :return: the rotated point
    '''
    @staticmethod
    def rotate_point(speed, angle):
        dx = speed * math.cos(angle + math.pi / 2)
        dy = speed * math.sin(angle + math.pi / 2)
        return np.array([dx, dy])

    '''
        Get the x position of the vacuum cleaner

        :return: the x position of the vacuum cleaner
    '''
    @property
    def x(self):
        return self._pos[0]

    '''
        Get the y position of the vacuum cleaner

        :return: the y position of the vacuum cleaner
    '''
    @property
    def y(self):
        return self._pos[1]

    '''
        Get the angle of the vacuum cleaner

        :return: the angle of the vacuum
    '''
    @property
    def angle(self):
        return self._angle

'''
    Calculate the squared distance between two points

    :param p1: the first point
    :param p2: the second point

    :return: the squared distance between the two points
'''
def sq_dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

'''
    Calculate the intersection between two lines

    :param line1: the first line
    :param line2: the second line

    :return: the intersection between the two lines
'''
def line_intersection(line1, line2):
    """Receives two lines defined by two points belonging to each. Returns their intersection"""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return float('inf'), float('inf')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y