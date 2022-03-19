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
csvdata= pd.read_csv("dataset.csv")
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

    print(plays)

'''
g = data[1]
g.portions = [{"AI" : 13.0/20.0, 'IOT' : 7.0/20.0}, {'AI' : 13.0/23.0, 'IOT' : 10.0/23}]
#strategy_profile = [{"Launch" : 0.5, 'DontLaunch' : 0.5}, {'Launch' : 0.1, 'DontLaunch' : 0.9}]
print(qlk(g, [0.46710301, 0.3740573,  0.15883968], [1., 1.00000036, 1.]))



#print(g.get_utility(0, strategy_profile))
#print(qbr(g, 0, strategy_profile, 1))
'''