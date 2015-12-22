"""Ranking Algorithms"""
from __future__ import division
import sys
import math
import numpy as np
import itertools
import glicko2 as g2


def weight(method, t, t0, tf):
    """ Evaluates a weight or scaling factor given a certain
            method and time. For example, 'linear' weights older
            games less than more recent games according to a
            linear distribution.
    """
    if method == None:
        return 1.0
    elif method == 'linear':
        return 1.0 + np.true_divide(t - t0, tf - t0)
    elif method == 'log':
        return 1.0 + np.log(1 + (np.true_divide(t - t0, tf - t0)))
    elif method == 'exponential':
        return np.exp(np.true_divide((t - t0), (tf - t0)))
    elif method == 'biweekly':
        return math.floor(((t - t0) / 14) + 1)
    else:
        print " Not a valid weighting method, therefore continuing unweighted."
        return 1.0


def massey(games, num_teams, team_dict, weight_method, criteria):
    """ Massey ranking method for teams given a set of games between
           them and a weight method. Criteria ranks according to point
           differential or win/loss or total yards differential.
    """
    # Time info (start, end) for weighting
    t0 = int(games[0].eid[:8])
    tf = int(games[-1].eid[:8])

    # T contains diagonal info about total games played
    T = np.zeros((num_teams, num_teams))
    # P contains off-diagonal info about pairwise matchups
    P = np.zeros((num_teams, num_teams))
    # b gives total yard differentials
    b = np.zeros(num_teams)
    for game in games:
        t = int(game.eid[:8])
        h = team_dict[game.home]
        a = team_dict[game.away]

        if (criteria == None) or (criteria == 'points'):
            h_yds = game.score_home 
            a_yds = game.score_away
        elif criteria == 'total_yards':
            h_yds = game.stats_home[1]
            a_yds = game.stats_away[1]
        elif criteria == 'pass_yards':
            h_yds = game.stats_home[2]
            a_yds = game.stats_away[2]
        elif criteria == 'rush_yards':
            h_yds = game.stats_home[3]
            a_yds = game.stats_away[3]
        else:
            print " Not a valid criteria!"
            # FIXME: better error handling

        if h_yds > a_yds:
            w = h
            l = a
            w_yds = h_yds
            l_yds = a_yds
        elif a_yds > h_yds:
            w = a
            l = h
            w_yds = a_yds
            l_yds = h_yds

        T[w, w] += 1.0 * weight(weight_method, t, t0, tf)
        T[l, l] += 1.0 * weight(weight_method, t, t0, tf)
        P[w, l] += 1.0 * weight(weight_method, t, t0, tf)
        P[l, w] += 1.0 * weight(weight_method, t, t0, tf)
        b[w] += (w_yds - l_yds) * weight(weight_method, t, t0, tf)
        b[l] += (l_yds - w_yds) * weight(weight_method, t, t0, tf)
    Massey_Matrix = T - P
    # Replace top row of Massey_Matrix with ones and corresponding
    # entry in p_vector to zero to ensure full rank
    Massey_Matrix[0, :] = np.ones(num_teams)
    b[0] = 0.0
    rating = np.linalg.solve(Massey_Matrix, b)
    return rating


def colley(games, num_teams, team_dict, weight_method, criteria):
    """ Colley ranking method for teams given a set of games between
           them and a weight method. Colley uses win/loss ONLY.
    """
    if criteria != None:
        print "Colley only uses win/loss, not points, or yards, etc"

    # Time info (start, end) for weighting
    t0 = int(games[0].eid[:8])
    tf = int(games[-1].eid[:8])

    Colley_Matrix = np.zeros((num_teams, num_teams))

    # Add two to the diagonal
    Colley_Matrix += 2 * np.eye(num_teams)
    b_vector = np.ones(num_teams)
    for game in games:
        # grab time of game, set to 't'
        t = int(game.eid[:8])
        h = team_dict[game.home]
        a = team_dict[game.away]
        h_pts = game.score_home
        a_pts = game.score_away
        if h_pts > a_pts:
            w = h
            l = a
        elif a_pts > h_pts:
            w = a
            l = h
        # need to determine how to handle ties!

        # Add a game played to team's diagonal element
        Colley_Matrix[w, w] += 1.0 * weight(weight_method, t, t0, tf)
        Colley_Matrix[l, l] += 1.0 * weight(weight_method, t, t0, tf)
        # Add negative one to team's off diagonal
        Colley_Matrix[w, l] += -1.0 * weight(weight_method, t, t0, tf)
        Colley_Matrix[l, w] += -1.0 * weight(weight_method, t, t0, tf)
        b_vector[w] += 0.5 * weight(weight_method, t, t0, tf)
        b_vector[l] += -0.5 * weight(weight_method, t, t0, tf)
    rating = np.linalg.solve(Colley_Matrix, b_vector)
    return rating
def Keener_skew(x):
    return 0.5 + 0.5*(np.sign(x - 0.5)*np.sqrt(abs(2.0*x - 1.0)))

def keener(games, num_teams, team_dict, weight_method, criteria):

    # Time info (start, end) for weighting
    t0 = int(games[0].eid[:8])
    tf = int(games[-1].eid[:8])

    epsilon = 1e-12
    # S_Matrix contains points scored against other teams 
    S_Matrix = np.zeros((num_teams,num_teams))
    a_Matrix = np.zeros((num_teams,num_teams))
    e_vector = np.ones(num_teams)
    games_played = np.zeros((num_teams,1))

    for game in games:
        t = int(game.eid[:8])
        h = team_dict[game.home]
        a = team_dict[game.away]

        if (criteria == None) or (criteria == 'points'):
            h_yds = game.score_home
            a_yds = game.score_away
        elif criteria == 'total_yards':
            h_yds = game.stats_home[1]
            a_yds = game.stats_away[1]
        elif criteria == 'pass_yards':
            h_yds = game.stats_home[2]
            a_yds = game.stats_away[2]
        elif criteria == 'rush_yards':
            h_yds = game.stats_home[3]
            a_yds = game.stats_away[3]
        else:
            print " Not a valid criteria!"
            # FIXME: better error handling
        w = h
        l = a
        w_yds = h_yds  
        l_yds = a_yds 

        # games played keeps track of how many games each team has played 
        # (e.g. bye weeks, etc.) so we can normalize at the end.
        games_played[w] += 1.0
        games_played[l] += 1.0
        # The S_Matrix contains the raw data. In this case, we do 
        # points Team i scored against Team j. We could easily do yards
        # gained or passes completed, etc.
        S_Matrix[w,l] += w_yds*weight(weight_method,t,t0,tf)
        S_Matrix[l,w] += l_yds*weight(weight_method,t,t0,tf)
        a_Matrix[w,l] = ((S_Matrix[w,l] + 1.0)/(S_Matrix[w,l] + S_Matrix[l,w] + 2.0))
        a_Matrix[l,w] = ((S_Matrix[l,w] + 1.0)/(S_Matrix[l,w] + S_Matrix[w,l] + 2.0))
    # apply keener's skewing function
    a_Matrix = Keener_skew(a_Matrix)
    # this division performs normalization routine. be careful! must be float.
    a_Matrix = np.divide(a_Matrix,games_played)
    a_Matrix = a_Matrix + epsilon*np.dot(e_vector,e_vector.T)
    E,R = np.linalg.eig(a_Matrix)
    idx = np.argmax(E)
    r = R[:,idx]
    r = r/np.sum(r) # renormalize
    return r

def reorder(games, num_teams, team_dict, weight_method, criteria, init): 
    #FIXME: not currently working

    # Time info (start, end) for weighting
    t0 = int(games[0].eid[:8])
    tf = int(games[-1].eid[:8])

    # S_Matrix contains points scored against other teams 
    S_Matrix = np.zeros((num_teams,num_teams))
    a_Matrix = np.zeros((num_teams,num_teams))
    e_vector = np.ones(num_teams)
    games_played = np.zeros((num_teams,1))

    for game in games:
        t = int(game.eid[:8])
        h = team_dict[game.home]
        a = team_dict[game.away]

        if (criteria == None) or (criteria == 'points'):
            h_yds = game.score_home
            a_yds = game.score_away
        elif criteria == 'total_yards':
            h_yds = game.stats_home[1]
            a_yds = game.stats_away[1]
        elif criteria == 'pass_yards':
            h_yds = game.stats_home[2]
            a_yds = game.stats_away[2]
        elif criteria == 'rush_yards':
            h_yds = game.stats_home[3]
            a_yds = game.stats_away[3]
        else:
            print " Not a valid criteria!"
            # FIXME: better error handling
        w = h
        l = a
        w_yds = h_yds  
        l_yds = a_yds 

        # games played keeps track of how many games each team has played 
        # (e.g. bye weeks, etc.) so we can normalize at the end.
        games_played[w] += 1.0
        games_played[l] += 1.0
        # The S_Matrix contains the raw data. In this case, we do 
        # points Team i scored against Team j. We could easily do yards
        # gained or passes completed, etc.
        S_Matrix[w,l] += w_yds*weight(weight_method,t,t0,tf)
        S_Matrix[l,w] += l_yds*weight(weight_method,t,t0,tf)
        a_Matrix[w,l] = ((S_Matrix[w,l] + 1.0)/(S_Matrix[w,l] + S_Matrix[l,w] + 2.0))
        a_Matrix[l,w] = ((S_Matrix[l,w] + 1.0)/(S_Matrix[l,w] + S_Matrix[w,l] + 2.0))
    # apply keener's skewing function
    a_Matrix = np.divide(a_Matrix,games_played)
    a_Matrix = a_Matrix + 0.00000000001*np.dot(e_vector,e_vector.T)
    D = a_Matrix 
    D = D/np.sum(D)
    def makeR(n):
        R = np.zeros((n,n))
        for i in range(n-1):
            R += (i+1)*np.eye(n,k=-(i+1))
        R = R/np.sum(R)
        return R
    R = makeR(num_teams)
    Q = np.eye(num_teams,num_teams)
    def fitness(a):
        return np.linalg.norm(np.dot(Q[a].T,np.dot(D,Q[a])) - R)
    
    def mate(a,b):
        rate = 0.4
        do_flip = np.random.random()
        if rate > do_flip:
            n = len(a)
            site1 = np.random.randint(0,n)
            site2 = np.random.randint(0,n)
    
            if site1 > site2:
                site2,site1 = site1,site2
    
            c = np.copy(b)
            chunk = a[site1:site2]
            c[site1:site2] = chunk
            leftover = [x for x in b if x not in chunk]
            c[:site1] = leftover[:len(c[:site1])]
            c[site2:] = leftover[len(c[:site1]):]
            return c
        else:
            return a  
    
    def mutate(a):
        m = np.copy(a)
        gene1 = np.random.randint(0,len(a)-1)
        gene2 = np.random.randint(0,len(a)-1)
        rate = 0.8
        do_flip = np.random.random()
        if rate > do_flip:
            m[gene1] = a[gene2]
            m[gene2] = a[gene1]
        return m
    
    def evolve(pop,it):
        total_fitness = 0.0
        pop_fit = []
        best_fitness = 100.0
        for idx, individual in enumerate(pop):
            this_fitness = fitness(individual)
            if this_fitness < best_fitness:
                best_fitness = this_fitness
                best_rank = individual
            total_fitness += this_fitness
            pop_fit.append(this_fitness)
        pop_sort = np.argsort(pop_fit)
        very_fit =  np.array(pop)[pop_sort]
        #fit =  np.array(pop)[pop_sort[:1]]
        #unfit =  np.array(pop)[pop_sort[1:]]
        #dead =  np.array(pop)[pop_sort[99:]]
            
        avg_fitness = total_fitness/idx 
    
        new_pop = []
        #for individual in unfit:
        #    new_pop.append(mutate(individual))
        #for individual in dead:
        #    new_pop.append(np.random.permutation(num_teams))
        #for individual in fit:
        #    partner = np.random.randint(0,len(fit))
        #    new_pop.append(mate(individual,fit[partner]))
        for individual in very_fit:
            new_pop.append(individual)
        if count%200 == 0:
            print "best = ", best_fitness
            print "average = ", avg_fitness
        return new_pop,best_rank,best_fitness
    
    
    
    population = []
    population.append(init)
    print "pop = ", population
    #for i in xrange(100):
    #    population.append(np.random.permutation(num_teams))
   
    best_fitness = 100   
    for count in xrange(1):
        population,this_best_rank,this_best_fitness = evolve(population,count)
        if this_best_fitness < best_fitness:
            best_fitness = this_best_fitness
            best_rank = this_best_rank
        
    r = np.array(best_rank)
    return r


def glicko2(games,num_teams,team_dict,criteria,rating,RD,vol):
    eps = 0.000000000001
    tau = 0.3 
    #rating = np.ones(num_teams)*1500
    #RD = np.ones(num_teams)*150
    #vol = np.ones(num_teams)*0.8
    # Create S matrix, P matrix
    Score_Matrix = np.zeros((num_teams,num_teams))
    s = np.zeros((num_teams,num_teams))
    p = np.zeros((num_teams,num_teams))
    e_vector = np.ones(num_teams)

    for game in games: 
        h = team_dict[game.home]
        a = team_dict[game.away]

        if (criteria is None) or (criteria == 'points'):
            h_yds = game.score_home 
            a_yds = game.score_away 
        elif criteria == 'total_yards':
            h_yds = game.stats_home[1]
            a_yds = game.stats_away[1]
        elif criteria == 'pass_yards':
            h_yds = game.stats_home[2]
            a_yds = game.stats_away[2]
        elif criteria == 'rush_yards':
            h_yds = game.stats_home[3]
            a_yds = game.stats_away[3]
        else:
            print " Not a valid criteria!"
            # FIXME: better error handling

        if h_yds >= a_yds:
            w = h
            l = a
            w_yds = h_yds
            l_yds = a_yds
        elif a_yds > h_yds:
            w = a
            l = h
            w_yds = a_yds
            l_yds = h_yds

        p[w,l] = 1.0
        p[l,w] = 1.0

        #Score_Matrix[w,l] += 2.0
        #Score_Matrix[l,w] += 1.0
        Score_Matrix[w,l] += w_yds
        Score_Matrix[l,w] += l_yds
                             
        s[w,l] = ((Score_Matrix[w,l] + 1.0)/(Score_Matrix[w,l] + Score_Matrix[l,w] + 2.0))
        s[l,w] = ((Score_Matrix[l,w] + 1.0)/(Score_Matrix[l,w] + Score_Matrix[w,l] + 2.0))
    #s = Keener_skew(s)
    #s  = s + 0.00000001*np.dot(e_vector,e_vector.T)
    rating,RD,vol,E = g2.main_update(rating,RD,vol,s,p,tau,eps)
    return rating,RD,vol


def main():
    """Main entry point for the script."""
    pass

if __name__ == '__main__':
    sys.exit(main())
