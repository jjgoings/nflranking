"""Ranking Algorithms"""
import sys
import math
import numpy as np


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
    if not criteria:
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


def main():
    """Main entry point for the script."""
    pass

if __name__ == '__main__':
    sys.exit(main())
