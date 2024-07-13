import random

# Blackjack simulation just seeing how long a strategy can maintain its stack of chips

# Default params
num_players = 4
num_decks = 5
starting_stack = 50

# Setting up the game
shoe = []
card_dict =  {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 'J':0, 'Q':0, 'K':0, 'A':0}
card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'] 
for i in range(num_decks):
    for c in card_types:
        shoe.append(c)
        card_dict[c] += 1
player_stacks = []
for player in range(num_players):
    player_stacks.append(starting_stack)

# Playing the game
# Shuffle the shoe
random.shuffle(shoe)