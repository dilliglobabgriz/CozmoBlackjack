# CozmoBlackjack

bjsimulator.py allows the user to run over 1,000,000 individual hands of blackjack with customizeable numbers of players, decks, strategies, and games. These files are saved to a CSV that can be modified using the CSV_NAME variable

useful_graphs.py runs analytics on the csvs created in the simulator. The current graph compares one of the important factors in the sims decision making, its odds of going over 21 when hitting, to its average win rate. In this case beating the dealer 100% of the time would have a value of 1, losing has a value of -1, and a push is 0
