from season import Season

season = Season()
season.year = 2014  # default season is 2015
season.massey()     # Massey ranking method, unweighted, criteria is point diffential
for team in season.rating:
    print team


# or, if you want, you can get fancier and weight the massey or colley

season.colley('log') # colley logarithmically weighted in time
season.massey('linear') # massey linearly weighted in time
season.massey('exponential','total_yards') # massey exponentially weighted in time, with total yards differential as the ranking criteria

