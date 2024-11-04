import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
from maps import *

MAX_STEER_DEG = int(np.rad2deg(MAX_STEER))
MAX_ACC = 2

front = ctrl.Antecedent(np.arange(0, MAX_DIST, 1), 'front')
vel = ctrl.Antecedent(np.arange(MAX_REV_SPEED, MAX_SPEED, 0.01), 'vel')
ang_vel = ctrl.Antecedent(np.arange(-MAX_STEER_DEG, MAX_STEER_DEG, 0.1), 'ang_vel')
acc = ctrl.Consequent(np.arange(-MAX_ACC, MAX_ACC, 0.01), 'acc')
ang_acc = ctrl.Consequent(np.arange(-1, 1, 0.01), 'ang_acc')

'''
    distance definitions
'''
front['close'] = fuzz.trimf(front.universe, [0, 25, 39])
front['medium'] = fuzz.trimf(front.universe, [40, 50, 59])
front['far'] = fuzz.trapmf(front.universe, [60, 70, MAX_DIST, MAX_DIST])

'''
    velocity definitions
'''
vel['stop'] = fuzz.trimf(vel.universe, [-0.01, 0, 0.01])

'''
    angular velocity definitions
'''
ang_vel['left'] = fuzz.gaussmf(ang_vel.universe, -MAX_STEER_DEG, 1)
ang_vel['none'] = fuzz.trimf(ang_vel.universe, [-MAX_STEER_DEG * 0.1, 0, MAX_STEER_DEG * 0.1])
ang_vel['right'] = fuzz.gaussmf(ang_vel.universe, MAX_STEER_DEG, 1)

'''
    acceleration definitions
'''
acc['none'] = fuzz.trimf(acc.universe, [-0.5, 0, 0.5])
acc['front'] = fuzz.trimf(acc.universe, [1, 2, 2])

'''
    angle acceleration definitions
'''
ang_acc['sharp_left'] = fuzz.gaussmf(ang_acc.universe, -1, 0.002)
ang_acc['left'] = fuzz.gaussmf(ang_acc.universe, -1, 0.020)
ang_acc['none'] = fuzz.gaussmf(ang_acc.universe, 0, 0.125)
ang_acc['right'] = fuzz.gaussmf(ang_acc.universe, 1, 0.020)
ang_acc['sharp_right'] = fuzz.gaussmf(ang_acc.universe, 1, 0.002)

# rules
rules = [
    #if the distance to the nearest obstacle in front of the vacuum cleaner is far, the vacuum cleaner should accelerate
    ctrl.Rule(~vel['stop'] & ang_vel['none'] & front['far'], acc['front']),
    #if the distance to the nearest obstacle in front of the vacuum cleaner is medium, the vacuum cleaner should turn left or right depending on the angular velocity
    ctrl.Rule(~front['medium'] & ang_vel['left'], ang_acc['right']),
    ctrl.Rule(~front['medium'] & ang_vel['right'], ang_acc['left']),
    #if the distance to the nearest obstacle in front of the vacuum cleaner is close, the vacuum cleaner should turn left or right depending on the angular velocity
    ctrl.Rule(front['close']  & ang_vel['left'] , ang_acc['sharp_left']),
    ctrl.Rule(front['close'] & ang_vel['right'] , ang_acc['sharp_right']),
    ctrl.Rule(front['close']  & ang_vel['none'] , ang_acc['sharp_left']),
    #if the velocity of the vacuum cleaner is stop and the distance to the nearest obstacle in front of the vacuum cleaner is close, the vacuum cleaner should accelerate
    ctrl.Rule(vel['stop'] & front['close'], acc['none']),
    ctrl.Rule(vel['stop'] & ~front['medium'], acc['front']),
    ctrl.Rule(vel['stop'] & ~front['far'], acc['front']),
    #if the velocity of the vacuum cleaner is stop and the distance to the nearest obstacle in front of the vacuum cleaner is close, the vacuum cleaner should not accelerate
    ctrl.Rule(~vel['stop'] & front['close'], acc['none']),
]

fuzzy_controller = ctrl.ControlSystem(rules)
controller = ctrl.ControlSystemSimulation(fuzzy_controller)