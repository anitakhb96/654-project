from math import exp
from math import log2, log
import random
from statistics import mean
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
        #print(lam, game.get_utility(player, action_profile), action_profile, game.name)
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
                #print(alpha[k])
                res[p][a] += (all_pi[k][p][a] * alpha[k])
    return res

def get_diff(params, game, func):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    strategy = func(game, alpha, lam)
    res = [{a : abs(game.portions[p][a] - strategy[p][a]) for a in game.action_sets[p]} for p in range(game.n_players)]
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

def get_params(game, func, n_params):
    alpha = ((0.001, 1) for i in range(int(n_params/2)))
    lam = ((0, 5) for i in range(int(n_params/2)))
    bnds = tuple(list(alpha) + list(lam))
    #bnds = ((0.001, 1), (0.001, 1), (0.001, 1), (0., None), (0.0, None), (0.0, None))
    cons = {'type': 'eq', 'fun': lambda x:  sum([x[i] for i in range(int(len(x) / 2))]) - 1}
    x0 = tuple([1.0/int(n_params/2.0) for i in range(int(n_params/2.0))] + [1 for i in range(int(n_params/2.0))])
    res = minimize(objective_function, x0, method='SLSQP', bounds=bnds, constraints=cons, args=(game, func))
    return res.x

def risk(params, games, func):
    res = 0
    for g in games:
        res += get_loss(params, g, func)
    return res / len(games)


def get_loss(params, game, func):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    pred = func(game, alpha, lam)
    loss = 0
    for p in range(game.n_players):
        for a in game.action_sets[p]:
            loss += (game.portions[p][a] - pred[p][a]) ** 2
    return loss  

def KL(params, game, func):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    pred = func(game, alpha, lam)
    l = [sum([game.portions[p][a] * log2(game.portions[p][a] / pred[p][a]) for a in game.action_sets[p]]) for p in range(game.n_players)]
    return l 

def cross_entropy(params, game, func):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    pred = func(game, alpha, lam)
    l = [sum([-game.portions[p][a] * log(pred[p][a]) for a in game.action_sets[p]]) for p in range(game.n_players)]
    return l 

def avg_cross_entropy(params, games, func):
    return mean(mean(cross_entropy(params, g, func)) for g in games)
    
def k_fold(games, n_params, func):
    k_weights = []
    k_params = []
    for i in range(len(games)):
        train = games[i]
        params = get_params(train, func, n_params)
        loss = 0
        for j in range (len(games)):
            if i != j:
                loss += mean(cross_entropy(params, games[j], func))
                #loss += get_loss(params, games[j], func)
        loss /= (len(games)-1)
        k_weights.append(1.0/loss)
        k_params.append(params)
    _sum = [0 for i in range(n_params)]
    for i in range(len(k_params)):
        for j in range(len(_sum)):
            _sum[j] += k_params[i][j]*k_weights[i]
    
    total_weights = sum(k_weights)
    avg = [x / total_weights for x in _sum]

    return avg

def qch_(game, alpha, lam):
    pi_0 = [{a : 1/len(game.action_sets[p]) for a in game.action_sets[p]} for p in range(game.n_players)]
    all_pi = [pi_0]
    for k in range(1, len(alpha)):
        sp = [{a: 0 for a in game.action_sets[p]} for p in range(game.n_players)]
        for j in range(len(all_pi)):
            sp = [{a: (sp[p][a]+ alpha[j]*all_pi[j][p][a])/sum(alpha[:len(all_pi)]) for a in game.action_sets[p]} for p in range(game.n_players)]
        print("!!!", sp)
        br = [qbr(game, i, sp, lam[k]) for i in range(game.n_players)]
        '''
        for j in range(len(all_pi)):
            m = qbr(game, i, all_pi[j], lam[k])
            pi_i_k = {a: (pi_i_k[a]+ m * alpha[j])/sum(alpha[:len(all_pi)]) for a in game.action_sets[i]}
        '''
        all_pi.append(br)

    res = [{a : 0 for a in game.action_sets[p]} for p in range(game.n_players)]
    for k in range(len(all_pi)):
        for p in range(len(res)):
            for a in game.action_sets[p]:
                res[p][a] += (all_pi[k][p][a] * alpha[k])
    return res

def objective_function(params, game, func):
    alpha = params[0 : int(len(params)/2)]
    lam = params[int(len(params)/2) : ]
    strategy = func(game, alpha, lam)
    loss = 0
    for i in range(len(strategy)):
        p = strategy[i]
        for a in p.keys():
            loss += (p[a] - game.portions[i][a]) ** 2
    return loss

def qch(game, alpha, lam):
    pi_0 = [{a : 1/len(game.action_sets[p]) for a in game.action_sets[p]} for p in range(game.n_players)]
    all_pi = [pi_0]
    for k in range(1, len(alpha)):
        sp = []
        for i in range(game.n_players):
            pi_i_k = {a:0 for a in game.action_sets[i]}
            for j in range(k):
                m = qbr(game, i, all_pi[j], lam[k])
                for a in pi_i_k.keys():
                    pi_i_k[a] += alpha[j] * m[a] / sum(alpha[:k])
            sp.append(pi_i_k)
        all_pi.append(sp)

    res = [{a : 0 for a in game.action_sets[p]} for p in range(game.n_players)]
    for k in range(len(all_pi)):
        for p in range(len(res)):
            for a in game.action_sets[p]:
                res[p][a] += (all_pi[k][p][a] * alpha[k])
    return res