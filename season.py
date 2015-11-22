"""NFL Season Class"""
import sys
import nflgame

class Season(object):

    def update_games(self):
        return nflgame.games(self._year,      \
                             week=self._week, \
                             kind=self._kind)

    def __init__(self):
        self._year     = 2015
        self._week     = None
        self._kind     = 'REG'
        self._teams    = nflgame.teams
        self._games    = self.update_games()

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

    @property
    def week(self):
        return self._week
    @week.setter
    def week(self,value):
        if self._week != value:
            self._week  = value
            self._games = self.update_games()

    @property
    def kind(self):
        return self._kind
    @kind.setter
    def kind(self,value):
        if self._kind != value:
            self._kind  = value
            self._games = self.update_games()

def main():
    """Main entry point for the script."""
    season = Season()
    season.year = 2015

if __name__ == '__main__':
    sys.exit(main())
