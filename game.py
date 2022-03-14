from math import exp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#our game
n_actions = (2, 2, 2)
n_players = len(n_actions)
utilities = dict()
utilities.update({(0, 0, 0): (1, 0, 3)})
utilities.update({(0, 0, 1): (-3, -2, 0)})
utilities.update({(0, 1, 0): (-2, 0, -1)})
utilities.update({(0, 1, 1): (-4, 0, 0)})
utilities.update({(1, 0, 0): (1, 2, -1)})
utilities.update({(1, 0, 1): (-2, 0, 0)})
utilities.update({(1, 1, 0): (-3, 0, -3)})
utilities.update({(1, 1, 1): (-4, 0, 0)})

#prisoner's dilemma
'''
n_actions = (2, 2)
n_players = len(n_actions)
utilities = dict()
utilities.update({(0, 0): (-1, -1)})
utilities.update({(0, 1): (-3, 0)})
utilities.update({(1, 0): (0, -3)})
utilities.update({(1, 1): (-2, -2)})
'''

#profile is a whole strategy profile for all players including the player's stategy
def qbr(player, profile, lam):
    l = []
    _profile = [x for x in profile]
    for n in range(n_actions[player]):
        exp_util = 1
        _profile[player] = [1 if n == j else 0 for j in range(n_actions[player])]
        for k in utilities.keys():
            pr = 1
            for p in range(len(k)):
                a = k[p]
                pr *= _profile[p][a]
            exp_util += (pr * utilities[k][player])
        l.append(exp_util)
    _res = [exp(lam * u) for u in l]
    res = [i / sum(_res) for i in _res]
    return res

def pi_l0():
    return [[1/n_actions[player] for i in range(n_actions[player])] for player in range(n_players)]

def pi_l1(player, lam1):
    return qbr(player, pi_l0(), lam1)

def pi_l2(player, lam1_2,lam2):
    profile = [pi_l1(_player, lam1_2) for _player in range(n_players)]
    return qbr(player, profile, lam2)

#alpha is a vector
def strategy(player, alpha, lam1, lam1_2, lam2):
    res = []
    s = [i*alpha[0] for i in pi_l0()[player]] 
    s1= [i*alpha[1] for i in pi_l1(player, lam1)] 
    s2= [i*alpha[2] for i in pi_l2(player, lam1_2, lam2)]
    for i in range(len(s)):
        res.append(s[i]+s1[i]+s2[i])
    return res

portions = [[14/23, 9/23], [13/23, 10/23], [9/23, 14/23]]

def obj_func(params):
    alpha0 = params[0]
    alpha1 = params[1]
    alpha2 = params[2]
    lam1 = params[3]
    lam1_2 = params[4]
    lam2 = params[5]
    s = [strategy(p, [alpha0, alpha1, alpha2], lam1, lam1_2, lam2) for p in range(n_players)]
    loss = sum([sum([(s[i][j] - portions[i][j]) ** 2 for j in range(len(s[i]))]) for i in range(n_players)])
    return loss

res = minimize(obj_func, [0.3, 0.3, 0.4, 1, 1, 1], method='Nelder-Mead', tol=1e-6)
print(res.x)          
#print(strategy(2, [0.05, 0.9, 0.05], 1, 1, 1))
print(strategy(0, [7.82608695e-01, 4.34782620e-02, 1.73913043e-01], 7.57117933e+01, -1.59331599e+02, 7.11405602e+01))