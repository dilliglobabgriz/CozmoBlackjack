#blackjack simulator
# Adapted from code by Denise Szecsei

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


def run_sim(players, num_decks, simulations, strategy):
    #first, let's make a shoe
    def make_shoe(num_decks, card_types):
        new_shoe = []
        for i in range(num_decks):
            for j in range(4):
                new_shoe.extend(card_types)
        random.shuffle(new_shoe)
        return new_shoe


    #next, we need a function to add up the value of the cards in a hand.  Aces can be 1 or 11, and we need to allow for both
    #a hand consists of a list of cards, and this function returns a list of the possible values of the total, depending on the
    #number of aces.  Because a shoe has several decks, we need to allow for the possibility of lots of aces.  If there is one
    #ace, the ace can be either 1 or 11.  two aces can add up to 2 or 12 (NOTE: only one ace in a hand can count as 11)
    #in general, k aces can add up to either k or k + 10, but we only care about an ace being 11 if it doesn't make the player go bust
    #as a naive approach, we will just consider the value of the ace to be the one that yields the highest hand value...this might change
    #in your decision-making strategy, especially if counting it as an 11 results in a bust, but counting it as a 1 keeps a player alive
    # Added feature that deals with hands of length 1
    def find_total(hand):
        face_cards = ['K', 'Q', 'J']
        aces = 0
        total = 0
        # Handle cases where hand is only one card (looking at dealer up card)
        # 3 cases are 2-10, aces, and face cards
        if type(hand) == type(1):
            return hand
        elif hand == 'A':
            return 11
        elif type(hand) == type('J'):
            return 10
        #a card will either be an integer in [2, 10] or a face card, or an ace
        for card in hand:
            if card == 'A':
                aces = aces + 1
            elif card in face_cards:
                total = total + 10
            else:
                total = total + card
            #at this point, we have the total of the hand, excluding any aces. 
        if aces == 0:
            return total
        else:
            #count all of the aces as 1, and return the highest of the possible total values, in ascending order; this is one place
            #where our approach could be improved
            ace_hand = [total + aces]
            ace_total = total + aces + 10
            if  ace_total < 22:
                ace_hand.append(ace_total)
            return max(ace_hand)

    # Function that returns the current "count" of the  deck based on how many face cards, aces, and low cards have been played
    # Takes card_count and num_decks as parameters
    # Uses the card_count_vals dictionary to calculate the count
    # Divide running count by the number of remaining deck to get the true count
    # Using griffin ultimate counting system
    card_count_vals = {2:1, 3:1, 4:1, 5:1, 6:1, 7:0, 8:0, 9:0, 10:-1, 'J':-1, 'Q':-1, 'K':-1, 'A':-1}
    def get_count(card_dict, decks):
        running_count = 0
        cards_remaining = 0
        for key in card_dict:
            cards_remaining += card_dict[key]
            running_count += (card_dict[key]*card_count_vals[key])
        # Return the true count
        # Calculated as running count/number of decks remaining in shoe
        # May switch the interger division for simplicity
        return running_count/(decks-(cards_remaining/52))

    # Should return the odds of busting if you hit with the inputted hand
    # Takes as input the current hand and the remaining dealer's cards (the shoe)
    # Dealer bust dict matches dealer up card to odds of busting
    # This is calculated by looking at dealer bust odds by card over large sample size
    dealer_bust_dict = {2:.34, 3:.35, 4:.40, 5:.43, 6:.42, 7:.27, 8:.24, 9:.24, 10:.22, 'J':.21, 'Q':.21, 'K':.22, 'A':.11}
    def bust_chance(hand, shoe):
        remaining_cards = 0
        bust_cards = 0
        # range(10) gives 0-9, i want 1-10
        for i in card_types:
            # If the card would make us bust add the total number of that card remaining in the deck to bust_cards
            if find_total(hand) + find_total(i) + 1 > 21:
                bust_cards += shoe[i]
                remaining_cards += shoe[i]
            # Whether or not we bust with the card, add it to the remaining cards to use as the divisor
            else:
                remaining_cards += shoe[i]
        return bust_cards/remaining_cards

    # Hit decision takes as input - current hand, dealers hand, and the "count" of the deck
    # Only has to account for hands with total values >= 12 
    # Returns as output odds, a float (0-1) 
    # If random.random() is greater than odds we will hit, so lower odds mean more hitting
    def hit_decision(player_hand, dealer_up_card, count):
        # Default
        odds = 0.5
        if find_total(player_hand) < 12:
            return 0
        # Break up into two cases, hands with ace and hands without
        # Calculation for hands without an ace
        # Based loosely on blackjack basic strategy cards
        if 'A' in player_hand:
            if find_total(player_hand) >= 19:
                odds = 1
            else:
                # As the player hand total is greater hit less often
                # Values should range from 0.1 to 0.9
                odds = 0.1 + (find_total(player_hand)-12) * 0.1
        else:
            # If hand total is more than 16 always stand
            if find_total(player_hand) >= 17:
                odds = 1
            else:
                # As the dealers up card is larger, we will hit more often
                # Could modify to use variable in order to run tests with different values here
                # Ace is 11 which is why values are off but should make odds range from 0.05 to 0.95
                odds = 1.15 - find_total(dealer_up_card) * 0.1

        return odds

    # Hit decision2 takes as input - current hand, dealers hand, and the "count" of the deck
    # Only has to account for hands with total values >= 12 
    # Returns as output odds, a float (0-1) 
    # If random.random() is greater than odds we will hit, so lower odds mean more hitting
    def hit_decision2(player_hand, dealer_up_card, count):
        # Default odds
        odds = 0.5

        pChance = bust_chance(player_hand, count)
        # Find difference between player busting odds and dealer bust odds
        diff_odds = pChance - dealer_bust_dict[dealer_up_card]
        # Never hit if odds of busting are over (x) percent
        # Also never hit if we are at hard 17+
        if ('A' not in hand) and (find_total(player_hand) < 17):
            odds = pChance
        # Never hit at soft 19+
        elif ('A' in hand) and (find_total(player_hand) < 19):
            odds = pChance
        else:
            odds = 1

        return odds

    # Hit decision 3 - basic strategy and card counting combined
    # Subtracts the difference between the player's chance of going over and the dealer's chance of going over to the odds
    # This means that is the dealer is much more likely to go over than the player, the player will stay more often (and vice versa)
    def hit_decision3(player_hand, dealer_up_card, count):
        # Default
        odds = 0.5
        
        # Difference between odds of going over
        diffOdds = bust_chance(player_hand, count) - dealer_bust_dict[dealer_up_card]

        if find_total(player_hand) < 12:
            return 0
        # Break up into two cases, hands with ace and hands without
        # Calculation for hands without an ace
        # Based loosely on blackjack basic strategy cards
        if 'A' in player_hand:
            if find_total(player_hand) >= 19:
                return 1
            else:
                # As the player hand total is greater hit less often
                # Values should range from 0.1 to 0.9
                odds = 0.1 + (find_total(player_hand)-12) * 0.1
        else:
            # If hand total is more than 16 always stand
            if find_total(player_hand) >= 17:
                return 1
            else:
                # As the dealers up card is larger, we will hit more often
                # Could modify to use variable in order to run tests with different values here
                # Ace is 11 which is why values are off but should make odds range from 0.05 to 0.95
                odds = 1.15 - find_total(dealer_up_card) * 0.1

        # Return the odds from the original hit decision formula - the odds difference divided by a factor of 5
        return odds - diffOdds/5


    #next, let's simulate ONE game, once the cards have been dealt...we will use this function to determine the player strategy
    #dealer_hand: 2 cards the dealer has
    #player_hands: the cards that the players have
    #curr_player_results: a list containing the result of each player's hand for this round; if there are three players, it might be [1, -1, 1]
    #dealer_cards: the cards left in the shoe; the shoe with the cards that have been dealt to the players for hitting will have been removed
    #hit_stay: is used to determine if a player hits or stays...you'll probably modify this in your own decision-making process
    #card_count: a dictionary to store the counts of the various card values that have been seen, for future card
    #counting in influencing our decision making and training data
    def play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, card_count, dealer_bust):
        #first, check if the dealer has blackjack.  that can only happen if the dealer has a total of 21, logically, and 
        #the game will be over before it really gets started...the players cannot hit
        if (len(dealer_hand) == 2) and (find_total(dealer_hand) == 21):
            for player in range(players):
                hit_count = 0
                curr_count = get_count(card_count, num_decks)
                #update live_action for the players, since they don't have a choice
                live_action.append(0)
                
                # Add dealer bust odds
                dealer_bust_odds.append(0.0)
                # Add the players bust odds based on starting hand to the csv
                player_bust_odds.append(bust_chance(player_hands[player], card_count))

                #check if any of the players also have blackjack, if so, they tie, and if not, they lose
                if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                    curr_player_results[0, player] = 0
                else:
                    curr_player_results[0, player] = -1    
                hit_count_list.append(hit_count)
                curr_count_list.append(round(curr_count, 1))

        #now each player can make their decisions...first, they should check if they have blackjack
        #for this player strategy, the decision to hit or stay is random if the total value is less than 12...
        #so it is somewhat unrelated to the cards they actually have been dealt (and is conservative), and ignores the card 
        #that the dealer has.  We will use this strategy to generate training data for a neural network.  
        #your job will be to improve this strategy, incorporate the dealer's revealed card, train a new neural
        #network based on that simulated data, and then compare the results of your neural network to the baseline
        #model generated from this training data.
        else:
            for player in range(players):
                hit_count = 0
                curr_count = get_count(card_count, num_decks)
                #the default is that they do not hit
                action = 0

                # Dealer bust odds list
                dealer_bust_odds.append(dealer_bust_dict[dealer_hand[1]])

                # Add the players bust odds based on starting hand to the csv
                player_bust_odds.append(round(bust_chance(player_hands[player], card_count), 2))
                
                #check for blackjack so that the player wins
                if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                    curr_player_results[0, player] = 1
                else:
                    # Instead of randomly choosing hit or stay the odds are skewed in favor one choice
                    # This is based on the hit_decision which replaced hit_stay
                    # Below 12 is always hitting because of no doubles
                    #hit_decision(player_hands[player], dealer_hand[1], get_count(card_count, num_decks))
                            
                    '''
                    Modify hit decision here
                    '''
                    if strategy == 'naive':
                        while (find_total(player_hands[player]) < 12):
                            hit_count += 1

                            #deal a card
                            player_hands[player].append(dealer_cards.pop(0))
                            
                            #update our dictionary to include the new card
                            card_count[player_hands[player][-1]] += 1
                            
                            #note that the player decided to hit
                            action = 1
                            
                            #get the new value of the current hand regardless of if they bust or are still in the game
                            #we will track the value of the hand during play...it was initially set up in the section below,
                            #and we are just updating it if the player decides to hit, so that it changes
                            live_total.append(find_total(player_hands[player]))                      
                                
                            #if the player goes bust, we need to stop this nonsense and enter the loss...
                            #we will record their hand value outside of the while loop once we know the player is done
                            if find_total(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break 
                    if strategy == 'no_count':
                        while (random.random() > hit_decision(player_hands[player], dealer_hand[1], get_count(card_count, num_decks))):
                            hit_count += 1

                            #deal a card
                            player_hands[player].append(dealer_cards.pop(0))
                            
                            #update our dictionary to include the new card
                            card_count[player_hands[player][-1]] += 1
                            
                            #note that the player decided to hit
                            action = 1
                            
                            #get the new value of the current hand regardless of if they bust or are still in the game
                            #we will track the value of the hand during play...it was initially set up in the section below,
                            #and we are just updating it if the player decides to hit, so that it changes
                            live_total.append(find_total(player_hands[player]))                      
                                
                            #if the player goes bust, we need to stop this nonsense and enter the loss...
                            #we will record their hand value outside of the while loop once we know the player is done
                            if find_total(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break  
                    if strategy == 'count_basic':
                        while (random.random() > hit_decision2(player_hands[player], dealer_hand[1], card_count)):
                            hit_count += 1
                            
                            #deal a card
                            player_hands[player].append(dealer_cards.pop(0))
                            
                            #update our dictionary to include the new card
                            card_count[player_hands[player][-1]] += 1
                            
                            #note that the player decided to hit
                            action = 1
                            
                            #get the new value of the current hand regardless of if they bust or are still in the game
                            #we will track the value of the hand during play...it was initially set up in the section below,
                            #and we are just updating it if the player decides to hit, so that it changes
                            live_total.append(find_total(player_hands[player]))                      
                                
                            #if the player goes bust, we need to stop this nonsense and enter the loss...
                            #we will record their hand value outside of the while loop once we know the player is done
                            if find_total(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break  
                    if strategy == 'count_smart':
                        while (random.random() > hit_decision3(player_hands[player], dealer_hand[1], card_count)):
                            hit_count += 1
                            
                            #deal a card
                            player_hands[player].append(dealer_cards.pop(0))
                            
                            #update our dictionary to include the new card
                            card_count[player_hands[player][-1]] += 1
                            
                            #note that the player decided to hit
                            action = 1
                            
                            #get the new value of the current hand regardless of if they bust or are still in the game
                            #we will track the value of the hand during play...it was initially set up in the section below,
                            #and we are just updating it if the player decides to hit, so that it changes
                            live_total.append(find_total(player_hands[player]))                      
                                
                            #if the player goes bust, we need to stop this nonsense and enter the loss...
                            #we will record their hand value outside of the while loop once we know the player is done
                            if find_total(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break  

                #update live_action to reflect the player's choice
                live_action.append(action)
                # Update hit count with the number of times each player hit
                hit_count_list.append(hit_count)
                curr_count_list.append(round(curr_count, 1))
                        
        #next, the dealer takes their turn based on the rules
        #first, the dealer will turn over their card, so we can count it and update our dictionary; this is the FIRST card they were dealt
        card_count[dealer_hand[0]] += 1
        
        while find_total(dealer_hand) < 17:
            #the dealer takes a card
            dealer_hand.append(dealer_cards.pop(0))    
            
            #update our dictionary for counting cards
            card_count[dealer_hand[-1]] += 1
        
        
        #this round is now complete, so we can determine the outcome...first, determine if the dealer went bust
        if  find_total(dealer_hand) > 21:
            
            #the dealer went bust, so we can append that to our tracking of when the dealer goes bust
            #we'll have to track the player outcomes differently, because if the dealer goes bust, a player
            #doesn't necessarily win or lose
            dealer_bust.append(1)
            
            #every player that has not busted wins
            for player in range(players):
                if curr_player_results[0, player] != -1:
                    curr_player_results[0, player] = 1
        else:
            #the dealer did not bust
            dealer_bust.append(0)
            
            #check if a player has a higher hand value than the dealer...if so, they win, and if not, they lose
            #ties result in a 0; for our neural network, we may want to lump ties with wins if we want a binary outcome
            for player in range(players):
                if find_total(player_hands[player]) > find_total(dealer_hand):
                    if find_total(player_hands[player]) < 22:
                        curr_player_results[0, player] = 1
                elif find_total(player_hands[player]) == find_total(dealer_hand):
                    curr_player_results[0, player] = 0
                else:
                    curr_player_results[0, player] = -1    
        
        #the hand is now complete, so we can return the results
        #we will return the results for each player
        return curr_player_results, dealer_cards, card_count, dealer_bust
                    

    #now we can run some simulations

    card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'] 
    special_cards = [10, 'J', 'Q', 'K'] 

    #set some parameters for the number of simulations (each simulation involves going through a shoe)
    #simulations = 3000

    #set the number of players in the game (we may want to set it to a value between 1 and 4?
    #players = 4

    #set the number of decks in a shoe.  this will impact the number of hands in a simulation
    #num_decks = 5

    #let's keep track of each round (dealer and player hands as well as the outcome) to analyze this data
    #there is no need to break this up by simulation, since what we want to analyze are the games, regardless
    #of which simulation it is in...but we will be able to track that information through the sim_number_list
    #if we wanted to analyze our data across simulations.

    #we want the cards that the dealer was dealt throughout the simulaton
    dealer_card_history = []
    dealer_total_history = []

    #we want all of the cards dealt to each player for each of the games in the simulation
    player_card_history = []

    #we want the player's outcome for each of the games in the simulation
    outcome_history = []

    #we want the hand values tracked for each of the games in the simulation
    player_live_total = []

    #we want to know whether the player hit during each of the games in the simulation
    player_live_action = []

    #we want to know if the dealer went bust in each of the games in the simulation
    dealer_bust = []

    #we need to keep track of our card counter throughout the simulation
    card_count_list = []

    # Keep track of current count for each hand
    curr_count_list = []

    # Create list for player and dealer odds of busting
    dealer_bust_odds = []
    player_bust_odds = []

    # Create hit count list for each player
    hit_count_list = []

    #we will track characteristics related to the shoe or simulation, as noted above:
    first_game = True
    prev_sim = 0
    sim_number_list = []
    new_sim = []
    games_played_in_sim = []


    #let's run our simulations

    for sim in range(simulations):
        
        #we aren't recording our data by simulation, but we could if we changed our minds
        #dealer_card_history_sim = []
        #player_card_history_sim = []
        #outcome_history_sim = []    
        
        games_played = 0
        
        #create the shoe
        dealer_cards = make_shoe(num_decks, card_types)
        
        #for each simulation, create a dictionary to keep track of the cards in the shoe, initially set to 0 for all cards
        card_count = {'A':0, 2:0, 3:0, 4:0, 5:0, 6:0,7:0, 8:0, 9:0, 10:0, 'J':0, 'Q':0, 'K':0}    
        
        #play until the shoe is almost empty...we can change this to be a function of the number of decks
        #in a shoe, but we won't start a game if there are fewer than 20 cards in a shoe...if we limit
        #the number of players to 4 (plus the dealer), then we'll need at least 10 cards for the game, and
        #we'll have enough cards for everyone to take 2...here's where the card counting could work to
        #a player's advantage
        while len(dealer_cards) > (players * 5 + 5):
            
            #here's how we will manage each game in the simulation:
            
            #keep track of the outcome of the players hand after the game: it will be 1, 0, -1
            curr_player_results = np.zeros((1, players))
            
            #create the lists for the dealer and player hands
            dealer_hand = []
            player_hands = [ [] for player in range(players)]
            live_total = []
            live_action = []

    
            
            #deal the FIRST card to all players and update our card counting dictionary
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
                card_count[player_hands[player][-1]] += 1
                
            #dealer gets a card, and the card counting dictionary is NOT updated
            dealer_hand.append(dealer_cards.pop(0))
            #card_count[dealer_hand[-1]] += 1
            
            #deal the SECOND card to all players and update our card counting dictionary
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
                card_count[player_hands[player][-1]] += 1
                
            #the dealer gets a card, and our card counter will be updated with the card that is showing
            dealer_hand.append(dealer_cards.pop(0))
            card_count[dealer_hand[-1]] += 1
            
            #record the player's live total after cards are dealt...if a player hits, we will update this information
            live_total.append(find_total(player_hands[player]))
            
            #flip a fair coin to determine if the player hits or stays...we can create a bias if we want to 
            #make this more sophisticated
            hit_stay = 0.5
            
            curr_player_results, dealer_cards, card_count, dealer_bust = play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, card_count, dealer_bust)
            
            #track the outcome of the hand
            #we want to know the dealer's card that is showing and their final total
            dealer_card_history.append(dealer_hand[1])
            dealer_total_history.append(find_total(dealer_hand))
            
            #this is the result of the hand for the players
            player_card_history.append(player_hands)
            
            #we want the outcome of each hand for each player
            outcome_history.append(list(curr_player_results[0]))
            
            #we want the evolution of each player's hand in a game, as well as whether they hit or not (this is 1 if the player ever hit)
            player_live_total.append(live_total)
            player_live_action.append(live_action)
            
            if sim != prev_sim:
                new_sim.append(1)
            else:
                new_sim.append(0)
                
            if first_game == True:
                first_game = False
            else:
                games_played += 1
            
            sim_number_list.append(sim)
            games_played_in_sim.append(games_played)
            card_count_list.append(card_count.copy())
            prev_sim = sim


    #create the dataframe for analysis.  My model will have the following features:
    #the dealer's second card is the one that is face up...
    #the player's initial hand value
    #whether the player hit or not
    #whether the dealer went bust or not
    #the dealer's total value

    #the outcome: win or lose

    # Modify dealer up card to be used multiple times
    dealer_card_history_split = []
    for card in dealer_card_history:
        for i in range(players):
            dealer_card_history_split.append(card)

    model_df = pd.DataFrame()

    # Add number of decks and number of players to database
    #model_df['num_players'] = players
    #model_df['num_decks'] = num_decks

    model_df['dealer_card'] = dealer_card_history_split
    #model_df['dealer_value'] = dealer_total_history

    #get initial hand values for all of the players and put them in the dataframe
    dealt_hand_values = []
    hand_list = []
    for i in range(len(player_card_history)):
        hands = player_card_history[i]
        for j in range(len(hands)):
            hand_list.append(find_total(hands[j][0:2]))
        
    model_df['player_initial_value'] = hand_list

    #get the action for each player for each game
    # Repeat each element in dealer bust 4 times so match 1 row per player
    player_live_action_flattened = [action for sublist in player_live_action for action in sublist]
    dealer_bust_split = []
    for b in dealer_bust:
        for i in range(players):
            dealer_bust_split.append(b)

    model_df['dealer_bust'] = dealer_bust_split

    #did the players win or lose? we will include a tie as a win for binary classification purposes
    # Flatten outcome history to match formatting
    outcome_history_flattened = [outcome for sublist in outcome_history for outcome in sublist]
    model_df['results'] = outcome_history_flattened

    # Split up hit so that it matches up with one row per player
    model_df['hit'] = player_live_action_flattened

    # Create the outcome column
    # If the player loses the round, the outcome should be the opposite of what the player did
    # If the player wins the round, their hit/stay decision is "correct"
    outcome_list = []
    for i in range(len(outcome_history_flattened)):
        temp_outcome = 0
        if (outcome_history_flattened[i] == 1 or outcome_history_flattened[i] == 0):
            temp_outcome = player_live_action_flattened[i]
        else:
            if player_live_action_flattened[i] == 1:
                temp_outcome = 0
            else:
                temp_outcome = 1
        outcome_list.append(temp_outcome)

    model_df['outcome'] = outcome_list

    model_df['p_bust_odds'] = player_bust_odds

    model_df['d_bust_odds'] = dealer_bust_odds

    model_df['hit_count'] = hit_count_list

    model_df['card_count'] = curr_count_list

    (model_df.info())
    print(model_df.describe())
    #write the data to a csv file, in case we want to refer to it later
    model_df.to_csv('blackjackdatacount_smart10k.csv')


# Sim customization and running
sim_version = input('Type 1 and press enter to run with default settings\nTo input your own parameters press 2:\n')

if sim_version == '1':
    run_sim(4, 5, 1000, 'naive')
elif sim_version == '2':
    get_players = input('Enter the number of players:\n')
    get_decks = input('Enter the number of decks:\n')
    get_simulations = input('Enter the number of simulations:\n')
    get_strategy = input('Enter your desired strategy - \nnaive, no_count, count_basic, count_smart:\n')
    while get_strategy not in ['naive','no_count','count_basic','count_smart']:
        print('That strategy is not available, please try again')
        get_strategy = input('Enter your desired strategy - \nnaive, no_count, count_basic, count_smart:\n')
    run_sim(int(get_players), int(get_decks), int(get_simulations), get_strategy)
else:
    print('This simulation version is not available')