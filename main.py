import numpy as np
import random

card_types = ['A',2,3,4,5,6,7,8,9,10,'J','Q','K']
special_cards = ['A','J','Q','K']
# Need decks in shoe, and num players
num_decks = 4
players = 3
# Number of simulations for data collection
simulations = 20

def make_shoe(num_decks, card_types):
    new_shoe = []
    for i in range(num_decks):
        for j in range(4):
            new_shoe.extend(card_types)
    # Shuffle the card
    random.shuffle(new_shoe)
    return new_shoe

# Function for getting value in hand
# Hand paramater is a list of cards
# Return is the value of the hand, aces will be 11 if possible
# Need to keep track of aces
# TODO: improve this component to give options for what to do with aces
def find_total(hand):
    face_cards = ['J','Q','K']
    aces = 0
    hand_value = 0
    # card can be ace(1,11), int(n), and face(10)
    # this loop adds the value of all cards besides the aces
    for card in hand:
        if card == 'A':
            aces += 1
        elif card in face_cards:
            hand_value += 10
        else:
            hand_value += card
    # hand the aces
    if aces == 0:
        return hand_value
    else:
        hand_value += aces
        if hand_value + 10 < 22:
            return (hand_value + 10)
        else:
            return hand_value
    
        
# Simulate a game
# Simulate one game once hands have been dealt
'''
dealer_hand: 2 cards dealer has
player_hands: array for the cards of the players hands
current_player_results: a numpy array of results of each players hand (WLT)
shoe: cards left in the shoe, when a player hits remove that card for this list
hit_stay: decision making function
card_count: dictionary to store cards removed from shoe
dealer_bust

NOTE: live_action keeps track of if player hits or stays during hand
'''
def play_hand(dealer_hand, player_hands, current_player_results, shoe, hit_stay, card_count, dealer_bust):
    # Check if dealer has blackjack
    if (len(dealer_hand) == 2) and (find_total(dealer_hand) == 21):
        # Dealer has blackjack, now check if any players push
        for player in range(players):
            # Update the live_action for the players
            live_action.append(0)

            # Check if any player also has blackjacl
            if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                current_player_results[0, player] = 0
            else:
                current_player_results[0, player] = -1

    # Dealer does not have blackjack
    # Player decision making
    # CHANGE THIS SECTION FOR MY TRAINING DATA
    else:
        for player in range(players):
            # Default as stay
            action = 0

            # Check for player having blackjack
            if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                # Player has blackjack
                current_player_results[0, player] = 1
            else:
                # No blackjack
                while (random.random() > hit_stay) and (find_total(player_hands[player]) < 14):
                    # Deal a card
                    player_hands[player].append(shoe.pop(0))

                    # Update the dicitionary for the card count
                    card_count[player_hands[player[-1]]] += 1

                    # Decided to hit so action should change to hit
                    action = 1

                    # Get the new value of current hand and update live action
                    live_total.append(find_total(player_hands[player]))

                    # See if player busts
                    if (find_total(play_hand[player]) > 21):
                        # The player loses
                        current_player_results[0, player] = -1
                        break
            # Update live action with player's chouce
            live_action.append(action)
        
    
    # Return the results for each player
    return current_player_results, shoe, card_count, dealer_bust

# Still need to deal with the dealer to finish sim
# Keep track of history
dealer_card_history = []
dealer_total_history = []
player_card_history = []
outcome_history = []
player_live_total = []
player_live_action = []
dealer_bust = []

card_count_list = []

# Track characteristics of shoe or sim
first_game = True
prev_sim = 0
sim_number_list = []
new_sim = []
games_played_in_sim = []

for sim in range(simulations):
    # Set up card counter dictionary
    card_count = {'A':0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 'J':0, 'Q':0, 'K':0}

    shoe = make_shoe(num_decks, card_types)

    # Decide when to shuffle with this
    while len(shoe) > 20:
        # Manage the game
        break

