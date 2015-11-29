"""NFL Season Class"""
import sys
import nflgame
import ranking
import numpy as np


class Season(object):

    def update_games(self):
        """ Update 'games' given new 'games' properties """
        return nflgame.games(self._year,
                             week=self._week,
                             kind=self._kind)

    def __init__(self,year=2015,week=None,kind='REG'):
        self._year = year   # Any year 2009 to present.
        self._week = week   # Specify individual weeks as integer.
        self._kind = kind   # kind = 'REG', 'POST', 'PRE' for reg. season, etc.
        self._teams = nflgame.teams
        self._games = self.update_games()
        self._num_teams = len(self._teams)
        self._num_games = len(self._games)
        # _team_dict maps team names to integer, and vice versa
        self._team_dict = {}
        for idx in xrange(self._num_teams):
            self._team_dict[idx] = str(self._teams[idx][0])
            self._team_dict[str(self._teams[idx][0])] = idx
        self._rating_vector = None
        self._rating = None
        self._RD = None
        self._vol = None

    @property
    def games(self):
        return self._games

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        if self._year != value:
            self._year = value
            self._games = self.update_games()
            self._num_games = len(self._games)

    @property
    def week(self):
        return self._week

    @week.setter
    def week(self, value):
        if self._week != value:
            self._week = value
            self._games = self.update_games()
            self._num_games = len(self._games)

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        if self._kind != value:
            self._kind = value
            self._games = self.update_games()
            self._num_games = len(self._games)

    @property
    def rating_vector(self):
        return self._rating_vector

    @property
    def num_teams(self):
        return self._num_teams

    @property
    def rating(self):
        result = []
        for i in xrange(self._num_teams):
            result.append([self._team_dict[i], self._rating_vector[i]])
        self._rating = sorted(result, key=lambda x: x[1], reverse=True)
        return self._rating

    def massey(self, weight=None, criteria=None):
        """ Wrapper for Massey rating method """
        self._rating_vector = ranking.massey(self._games,
                                             self._num_teams,
                                             self._team_dict,
                                             weight,
                                             criteria)

    def colley(self, weight=None, criteria=None):
        """ Wrapper for Colley rating method """
        # note criteria is useless for colley --- always uses win/loss data
        self._rating_vector = ranking.colley(self._games,
                                             self._num_teams,
                                             self._team_dict,
                                             weight,
                                             criteria)

    def keener(self, weight=None, criteria=None):
        """ Wrapper for Keener rating method """
        self._rating_vector = ranking.keener(self._games,
                                             self._num_teams,
                                             self._team_dict,
                                             weight,
                                             criteria)

    def glicko2(self, criteria=None,rating_vector=None,RD=None,vol=None):
        """ Wrapper for Glicko2 rating method """
        if rating_vector is None:
            rating_vector = np.ones(self._num_teams)*1500
        if RD is None:
            RD = np.ones(self._num_teams)*250
        if vol is None:
            vol = np.ones(self._num_teams)*0.08
        self._rating_vector,self._RD,self._vol = ranking.glicko2(self._games,
                                                     self._num_teams,
                                                     self._team_dict,
                                                     criteria,
                                                     rating_vector,
                                                     RD,
                                                     vol)


def main():
    """Main entry. Prints ranked teams in order."""
    season = Season()
    season.massey('log', 'total_yards')
    for team in xrange(season.num_teams):
        print season.rating[team]

if __name__ == '__main__':
    sys.exit(main())
