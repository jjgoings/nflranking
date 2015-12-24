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

def glicko(games,num_teams,team_dict,criteria,rating,RD):

    # helper functions
    def g(q,RD):
        return 1./np.sqrt(1. + 3.*q*q*RD*RD/(np.pi*np.pi))   

    def E(q,r,rj,RDj):
        return 1./(1. + np.power(10.,-g(q,RDj)*(r-rj)/400.0))

    def d_sq(q,r,outcomes):
        dsq = 0.0
        for outcome in outcomes:
            rj  = outcome[0]
            RDj = outcome[1]
            dsq += g(q,RDj)*g(q,RDj)*E(q,r,rj,RDj)*(1.0 - E(q,r,rj,RDj))
        dsq = dsq*q*q
        dsq = 1.0/dsq
        return dsq
    
    # system constants
    num_games = len(games)
    q = np.log(10.0)/400.0

    # initialization
    new_rating = np.copy(rating)
    new_RD     = np.copy(RD)

    for team in xrange(num_teams):
        ri       = rating[team]
        RDi      = RD[team]
        outcomes = []
        for game in games:

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

            if team == h:
                opponent = a 
                score    = (h_yds + 1.0)/(a_yds + h_yds + 2.0)
                outcomes.append([rating[opponent],RD[opponent],score])
            elif team == a:
                opponent = h 
                score    = (a_yds + 1.0)/(a_yds + h_yds + 2.0)
                outcomes.append([rating[opponent],RD[opponent],score])
        if outcomes:
            delta_ri = 0.0
            dsq      = d_sq(q,ri,outcomes)
            denom    = (1.0/(RDi*RDi) + 1.0/dsq)
            
            for outcome in outcomes:
                rj  = outcome[0]
                RDj = outcome[1]
                sj  = outcome[2]
                delta_ri += g(q,RDj)*(sj - E(q,ri,rj,RDj))
            delta_ri     *= q/denom

            rating[team] += delta_ri
            new_RD[team]  = np.sqrt(1.0/denom)
            
    return rating,new_RD


def glicko2(games,num_teams,team_dict,criteria,rating,RD,vol):
    eps = 0.0000000001
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

        # the += allows for multiple games played against the same opponent
        # although this rarely happens given how fast the program updates
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
