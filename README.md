This is an attempt to re-do several popular ranking algorithms for the NFL. 
I was getting increasingly unsatisfied with my old program, because it was not object oriented and very difficult to modify and test.

Right now I have (weighted) Colley and Massey methods, but plan to extend a bit more as I go. I'm still thinking about the overall structure, so feel free to open a new issue with suggestions. Feel free to fork if interested, but I'm in no rush.

As an example, here is how you could do an unweighted Massey rating of the 2014 NFL season. 

```python
from season import Season

season = Season()
season.year = 2014  # default season is 2015
season.massey()     # Massey ranking method, unweighted
for team in season.rating:
    print team
```

Which gives the output 

```
['NE', 11.407605241747872]
['DEN', 9.6038904227178108]
['SEA', 9.5169656599013628]
['GB', 8.2526935620363258]
...
['OAK', -8.9899785292554917]
['TB', -9.8107991356959232]
['JAC', -10.427123019821691]
['TEN', -11.805248019821692]

```


### Installation & Dependencies
You can simply clone this to your computer. As long as the `season.py` and `ranking.py` are in the same folder, you can `import season` and test it out. `ranking.py` simply contains helper ranking and rating functions for the `Season` class.

I have only tested with `Python2.7`.

Aside from the `numpy` dependencies (you *do* have `numpy`, don't you?), the only other package you should need is `nflgame`:
I use the [nflgame] (https://github.com/BurntSushi/nflgame) to import data.

Assuming you have the `pip` installer, you can simply do:
```
pip install nflgame
```


