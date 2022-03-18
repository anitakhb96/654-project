from math import exp
import random
from scipy.optimize import minimize


def get_action_profiles(payoffs):
    ls = dict()
    generate_action_profiles(payoffs, [], ls)
    return ls

def generate_action_profiles(dd, l, ls):
    if type(dd) == dict:
        for k in dd.keys():
            generate_action_profiles(dd[k], l+[k], ls)
    else:
        ls.update({tuple(l) : tuple(dd)})


class Game():
    def __init__(self, dic) -> None:
        self.dic = dic
        self.outcomes = get_action_profiles(dic['payoffs'])
        self.name = dic['game_name']
        self.n_players = len(list(self.outcomes.keys())[0])
        self.action_sets = self.get_actions_sets()
    
    def get_actions_sets(self):
        res = []
        d = self.dic['payoffs'].copy()
        while type(d) == dict:
            res.append(list(d.keys()))
            d = list(d.values())[0]
        return res
    
    def get_utility(self, player, strategy_profile):
        utility = 0
        profile_dist = dict()
        for profile in self.outcomes.keys():
            prob = 1
            for p in range(len(profile)):
                prob *= strategy_profile[p][profile[p]]
            utility += (prob * self.outcomes[profile][player])
        return utility

            

def qbr(game, player, strategy_profile, lam):
    utilities = dict()
    action_set = game.action_sets[player]
    for action in action_set:
        action_profile = [{a : 1 if a == action else 0 for a in action_set} if p == player else strategy_profile[p] for p in range(game.n_players)]
        numerator = exp(lam * game.get_utility(player, action_profile))
        utilities.update({action : numerator})
    denom = sum(utilities.values())
    utilities = {a : utilities[a] / denom for a in utilities.keys()}
    return utilities


def qlk(game, alpha, lam):
    pi_0 = [{a : 1/len(game.action_sets[p]) for a in game.action_sets[p]} for p in range(game.n_players)]
    all_pi = [pi_0]
    for k in range(1, len(alpha)):
        sp = []
        for i in range(game.n_players):
            pi_i_k = qbr(game, i, all_pi[-1], lam[k])
            sp.append(pi_i_k)
        all_pi.append(sp)

    res = [{a : 0 for a in game.action_sets[p]} for p in range(game.n_players)]
    for k in range(len(all_pi)):
        for p in range(len(res)):
            for a in game.action_sets[p]:
                res[p][a] += (all_pi[k][p][a] * alpha[k])
    return res

def qlk_objective_function(params, game):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    strategy = qlk(game, alpha, lam)
    loss = 0
    for i in range(len(strategy)):
        p = strategy[i]
        for a in p.keys():
            loss += (p[a] - game.portions[i][a]) ** 2
    return loss


def get_params(game):
    bnds = ((0, 1), (0, 1), (0, 1), (0, None), (0, None), (0, None))
    cons = {'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - 1}
    res = minimize(qlk_objective_function, (0.5, 0.25, 0.25, 1, 1, 1), method='SLSQP', bounds=bnds, constraints=cons, args=(game))
    return res.x

def get_loss(params, game):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    pred = qlk(game, alpha, lam)
    loss = 0
    for p in range(game.n_players):
        for a in game.action_sets[p]:
            loss += (game.portions[p][a] - pred[p][a]) ** 2
    return loss  
    
def k_fold(games, n_params):
    k_weights = []
    k_params = []
    for i in range(len(games)):
        train = games[i]
        params = get_params(train)
        loss = 0
        for j in range (len(games)):
            if i != j:
                loss += get_loss(params, games[j])
        loss /= (len(games)-1)
        k_weights.append(1.0/loss)
        k_params.append(params)
    sum = [0 for i in range(n_params)]
    for i in range(len(k_params)):
        for j in range(len(sum)):
            sum[j] += k_params[i][j]*k_weights[i]
    
    total_weights = sum(k_weights)
    avg = [x / total_weights for x in sum]

    return avg