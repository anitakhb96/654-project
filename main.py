import json

from utils import *
from scipy.optimize import minimize
import pandas as pd


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
params = k_fold(data[:-n_test], 6, qch)
print(params)
print('--------------')
print(risk(params, data[-n_test:], qch))

