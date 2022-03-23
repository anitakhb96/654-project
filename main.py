import json

from numpy import var
from sklearn.utils import shuffle

from utils import *
from scipy.optimize import minimize
import pandas as pd
import random


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
 
# Opening JSON file
json_file = open(dir_path + '/games.json')
data = json.load(json_file)
data =[Game(g) for g in data]
csvdata= pd.read_csv(dir_path + "/dataset.csv")
#print(csvdata['game_name'][0])


for g in data:
    plays = [{a : 0 for a in g.action_sets[i]} for i in range(g.n_players)]
    for p in range(g.n_players):
        for a in g.action_sets[p]:
            x = csvdata[(csvdata['game_name']==g.name)&(csvdata['player']==p)&(csvdata['action']==a)]['num_plays']
            plays[p][a] = int(x)
    for i in range(len(plays)):
        sum_ = sum(plays[i].values())
        plays[i] = {j: plays[i][j]/sum_ for j in plays[i].keys() }
    g.portions = plays





n_test = 5
avg = []
vqlk = [0, 0, 0, 0]
vqch = [0, 0, 0, 0]

#these games have been chosen randomly once
test_set = [
    '3player'
    'Felix-Narun-Betty-Ehsan-ZeroSum',
    'Felix-Narun-Betty-Ehsan-GeneralSum',
    'Bharathvaj-Taylor-Dominik-GeneralSum',
    'Alireza-Jordan-Kushagra-Tao-ZeroSum',
]

train_set = [g for g in data if g.name not in test_set]
test_set = [g for g in data if g.name in test_set]

'''
params = k_fold(train_set, 6, qlk)
print(params)
#print(qlk(data[-1], params[0:3], params[3:]))
print(mean([mean(cross_entropy(params, g, qlk)) for g in train_set]))
params = k_fold(train_set, 6, qch)
print(params)
#print(qlk(data[-1], params[0:3], params[3:]))
print(mean([mean(cross_entropy(params, g, qch)) for g in train_set]))
'''

'''
for i in range(50):
    print(i)
    random.shuffle(data)
    _data = data[:16]
    _data = shuffle(_data)
    val_set = _data[12:]
    test_set = data[16:]
    params = k_fold(train_set, 6, qlk)
    vqlk.append(avg_cross_entropy(params, val_set, qlk))
    params = k_fold(train_set, 6, qch)
    vqch.append(avg_cross_entropy(params, val_set, qch))

print(var(vqlk))
print(var(vqch))


'''
'''
ks = [4, 6, 8, 10]
for i in range(100):
    print(i)
    _data = data[:16]
    _data = shuffle(_data)
    train_set = _data[:12]
    val_set = _data[12:]
    ind = 0
    for ind in range(len(ks)):
        k = ks[ind]
        params = k_fold(train_set, k, qlk)
        vqlk[ind] += avg_cross_entropy(params, val_set, qlk)
        params = k_fold(train_set, k, qch)
        vqch[ind] += avg_cross_entropy(params, val_set, qch)

print(vqlk)
print(vqch)
print(var(vqlk))
print(var(vqch))
'''

#test
'''
for g in test_set:
    print(g.name)
    print("true: ", g.portions)
    params = k_fold(train_set, 6, qlk)
    alpha = params[:3]
    lam = params[3:]
    print('QLk: ', qlk(g, alpha, lam))
    print("CE: ", cross_entropy(params, g, qlk))
    params = k_fold(train_set, 6, qch)
    alpha = params[:3]
    lam = params[3:]
    print('QCH: ', qch(g, alpha, lam))
    print("CE: ", cross_entropy(params, g, qch))
    print('------------')
'''
params = k_fold(train_set, 6, qlk)
#print(qlk(test_set[0], params[:3], params[3:]))
print('qlk avg cross-entropy: ',avg_cross_entropy(params, test_set, qlk))
params = k_fold(train_set, 6, qch)
#print(qch(test_set[0], params[:3], params[3:]))
print('qlk avg cross-entropy: ', avg_cross_entropy(params, test_set, qch))