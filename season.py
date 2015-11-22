"""NFL Season Class"""
import sys
import nflgame
import ranking

class Season(object):

    def update_games(self):
        return nflgame.games(self._year,      \
                             week=self._week, \
                             kind=self._kind)

    def __init__(self):
        self._year      = 2015
        self._week      = None
        self._kind      = 'REG'
        self._teams     = nflgame.teams
        self._games     = self.update_games()
        self._num_teams = len(self._teams)
        self._num_games = len(self._games)
        # map team names to integer, and vice versa
        self._team_dict = {}
        for idx in xrange(self._num_teams):
            self._team_dict[idx] = str(self._teams[idx][0])
            self._team_dict[str(self._teams[idx][0])] = idx
        self._rating_vector = None
        self._rating        = None

    @property
    def games(self):
        return self._games

    @property
    def year(self):
        return self._year
    @year.setter
    def year(self,value):
        if self._year != value:
            self._year  = value
            self._games = self.update_games()
            self._num_games = len(self._games)

    @property
    def week(self):
        return self._week
    @week.setter
    def week(self,value):
        if self._week != value:
            self._week  = value
            self._games = self.update_games()
            self._num_games = len(self._games)

    @property
    def kind(self):
        return self._kind
    @kind.setter
    def kind(self,value):
        if self._kind != value:
            self._kind  = value
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
            result.append([self._team_dict[i],self._rating_vector[i]])
        self._rating = sorted(result,key=lambda x: x[1],reverse=True)
        return self._rating


    def massey(self,weight=None,criteria=None):
        self._rating_vector = ranking.massey(self._games, \
                                             self._num_teams, \
                                             self._team_dict, \
                                             weight, \
                                             criteria)

    def colley(self,weight=None,criteria=None):
        # note criteria is useless for colley --- always uses win/loss data
        self._rating_vector = ranking.massey(self._games, \
                                             self._num_teams, \
                                             self._team_dict, \
                                             weight, \
                                             criteria)


def main():
    """Main entry point for the script."""
    season = Season()
    season.massey('log','total_yards')
    for team in xrange(season.num_teams):
        print season.rating[team]

if __name__ == '__main__':
    sys.exit(main())
