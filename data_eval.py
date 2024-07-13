# Importing and handling data
# Credit to Dylan Fair 

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

#,num_decks,num_players,dealer_card,dealer_value,player_initial_value,hit,dealer_bust,results,count
def fully_evaluate_data(num_players = 4, num_decks = 5):
    df = pd.read_csv('blackjack/sims_csvs/blackjackdatacount_smart10k.csv')

    #Use a dictionary to keep tract of how the player does based on inital hand value
    # Each card has a list where in order the elements are the total/wins/losses/pushes
    initial_hand_dict = { 4: [0, 0, 0, 0], 5: [0, 0, 0, 0], 6: [0, 0, 0, 0], 7: [0, 0, 0, 0], 8: [0, 0, 0, 0], 9: [0, 0, 0, 0],
                        10: [0, 0, 0, 0], 11: [0, 0, 0, 0], 12: [0, 0, 0, 0], 13: [0, 0, 0, 0], 14: [0, 0, 0, 0], 15: [0, 0, 0, 0],
                        16: [0, 0, 0, 0], 17: [0, 0, 0, 0], 18: [0, 0, 0, 0], 19: [0, 0, 0, 0], 20: [0, 0, 0, 0], 21: [0, 0, 0, 0]}
    #Use a dictionary to keep tract of how the player does vs what a dealer's show card is
    # Each card has a list where in order the elements are the total/wins/losses/pushes
    dealer_card_dict = {'A': [0, 0, 0, 0], '2': [0, 0, 0, 0], '3': [0, 0, 0, 0], '4': [0, 0, 0, 0], '5': [0, 0, 0, 0],
                        '6': [0, 0, 0, 0], '7': [0, 0, 0, 0], '8': [0, 0, 0, 0], '9': [0, 0, 0, 0],
                        '10': [0, 0, 0, 0], 'J': [0, 0, 0, 0], 'Q': [0, 0, 0, 0], 'K': [0, 0, 0, 0]}
    num_wins = 0
    num_losses = 0
    num_pushes = 0
    dealer_bust_by_card = {'2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, 'J':0, 'Q':0, 'K':0, 'A':0}
    num_dealer_up_cards = {'2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, 'J':0, 'Q':0, 'K':0, 'A':0}
    total_hands = df.shape[0]
    for index, row in df.iterrows():
        if row['results'] == 1:
            num_wins += 1
            initial_hand_dict[row['player_initial_value']][0] += 1 
            initial_hand_dict[row['player_initial_value']][1] += 1
            dealer_card_dict[row['dealer_card']][0] +=1
            dealer_card_dict[row['dealer_card']][1] +=1
        elif row['results'] == -1:
            num_losses += 1
            initial_hand_dict[row['player_initial_value']][0] += 1 
            initial_hand_dict[row['player_initial_value']][2] += 1
            dealer_card_dict[row['dealer_card']][0] +=1
            dealer_card_dict[row['dealer_card']][2] +=1
        else:
            num_pushes +=1
            initial_hand_dict[row['player_initial_value']][0] += 1 
            initial_hand_dict[row['player_initial_value']][3] += 1
            dealer_card_dict[row['dealer_card']][0] +=1
            dealer_card_dict[row['dealer_card']][3] +=1

        dealer_bust_by_card[row['dealer_card']] += row['dealer_bust']
        num_dealer_up_cards[row['dealer_card']] += 1



    win_percent = round((num_wins/total_hands) * 100, 2)
    push_percent = round((num_pushes/total_hands) * 100, 2)
    loss_percent = round((num_losses/total_hands) * 100, 2)

    print(f"win percentage = {win_percent}, push percentage = {push_percent}, loss percentage = {loss_percent}")
    print('')

    
    # Create a plot that compares dealer up card to dealer bust percentages
    up_cards = list(dealer_bust_by_card.keys())
    up_cards_bust_percentage = []
    
    # Fill up_card_bust_percentage with the percentage of the time the dealer busts with each up card
    for i in up_cards:
        bust_odds = round((dealer_bust_by_card[i])/(num_dealer_up_cards[i]), 4) * 100
        #print(f'Dealer bust odds for {i} up: {bust_odds}%')
        up_cards_bust_percentage.append(bust_odds)
    plt.bar(up_cards, up_cards_bust_percentage)
    plt.ylabel('Percent')
    plt.title('Dealer bust percentage by dealer up card')
    plt.show()
    
    '''
    # Define plot space
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define x and y axes
    ax.bar(up_cards, 
        up_cards_bust_percentage,
        color = 'magenta')

    # Set plot title and axes labels
    ax.set(title = "Dealer Bust Odds\n(by up card)",
       xlabel = "Card",
       ylabel = "Odds of going over")

    plt.show()
    '''

    #Outputs a plot of the win/loss/push percentages
    names = ['Win', 'Loss', 'Push']
    bars = plt.bar(names, [win_percent, loss_percent, push_percent])

    # Change colors of bars
    bars[0].set_color('green')
    bars[1].set_color('red')
    bars[2].set_color('cyan')

    # Label bars
    for bar, label in zip(bars, [f"{win_percent}%", f"{loss_percent}%", f"{push_percent}%"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, label, ha='center', color='black')

    plt.ylabel(f'(%)')
    plt.title(f"Win, Loss, and Tie Percentages")
    plt.show()

    #Storing data from the plot
    win_dealer = []
    loss_dealer = []
    push_dealer = []
    card_dealer = []
    for key in dealer_card_dict.keys():
        card_dealer.append(key)
        total = dealer_card_dict[key][0]
        wins = dealer_card_dict[key][1]
        loss = dealer_card_dict[key][2]
        push = dealer_card_dict[key][3]
        player_wins = round((wins/total)*100, 2)
        win_dealer.append(player_wins)
        player_losses = round((loss/total) *100,2)
        loss_dealer.append(player_losses)
        player_pushes = round((push/total) *100,2)
        push_dealer.append(player_pushes)
        #print(f'When the dealer has {key} the player wins {player_wins} percent of hands, loses {player_losses} percent, and pushes {player_pushes} percent')
    print('')

    #Outputs a plot for how a player does vs a dealers initial card
    X_axis = np.arange(len(card_dealer))
    plt.bar(X_axis - 0.2, win_dealer, 0.2, label = 'Win')
    plt.bar(X_axis + 0, loss_dealer, 0.2, label = 'Losses')
    plt.bar(X_axis + 0.2, push_dealer, 0.2, label = 'Push')

    plt.xticks(X_axis, card_dealer)
    plt.xlabel("Dealers Card Showing")
    plt.ylabel("Percent win/loss/tie")
    plt.title("Player vs dealers card showing")
    plt.legend()
    plt.autumn()
    plt.show()

    #Storing data for how the player does vs their initial value
    win_player = []
    loss_player = []
    push_player = []
    value_player = []
    for key in initial_hand_dict.keys():
        total = initial_hand_dict[key][0]
        wins = initial_hand_dict[key][1]
        loss = initial_hand_dict[key][2]
        push = initial_hand_dict[key][3]
        value_player.append(key)
        if wins == 0:
            winPercent = 0
        else:
            winPercent = round((wins/total)*100, 2)
        if loss == 0:
            lossPercent = 0
        else:
            lossPercent = round((loss/total) *100,2)
        if push == 0:
            pushPercent = 0
        else: 
            pushPercent = round((push/total) *100,2)
        win_player.append(winPercent)
        loss_player.append(lossPercent)
        push_player.append(pushPercent)
        #print(f'When the player has {key} the player wins {winPercent} percent of hands, loses {lossPercent} percent, and pushes {pushPercent} percent')
    
    #Outputs a plot for how the player does vs their initial hand value
    X_axis = np.arange(len(value_player))
    plt.bar(X_axis - 0.2, win_player, 0.2, label = 'Win')
    plt.bar(X_axis + 0, loss_player, 0.2, label = 'Losses')
    plt.bar(X_axis + 0.2, push_player, 0.2, label = 'Push')

    plt.xticks(X_axis, value_player)
    plt.xlabel("Players Initial Vlaue")
    plt.ylabel("Percent win/loss/tie")
    plt.title("Player vs their Initial value")
    plt.legend()
    plt.show()


def evaluate_win():
    df = pd.read_csv('blackjackdata.csv')

    num_wins = 0
    num_losses = 0
    num_pushes = 0
    total_hands = df.shape[0]
    for index, row in df.iterrows():
        if row['results'] == 1:
            num_wins += 1
        elif row['results'] == -1:
            num_losses += 1
        else:
            num_pushes +=1

    win_percent = round((num_wins/total_hands) * 100, 2)
    push_percent = round((num_pushes/total_hands) * 100, 2)
    loss_percent = round((num_losses/total_hands) * 100, 2)

    print(f"win percentage = {win_percent}, push percentage = {push_percent}, loss percentage = {loss_percent}")

fully_evaluate_data()