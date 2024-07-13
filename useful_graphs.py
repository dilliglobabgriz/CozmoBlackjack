import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Config
strategy = 'count_smart'
simulations = '10k'

try:
    df = pd.read_csv(f'blackjack/sims_csvs/blackjackdata{strategy}{simulations}.csv')
except FileNotFoundError:
    print(f'The CSV file blackjackdata{strategy}{simulations}.csv was not found.\n')

def win_vs_p_odds():
    p_odds_buckets = ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.5', '.5-.6', '.6-.7', '.7-.8', '.8-.9', '.9-1']
    # Keeps track of number of wins and number of hands with these starting odds
    p_odds_dict = {i: [0, 0] for i in range(len(p_odds_buckets))}

    for index, row in df.iterrows():
        player_initial_value = row['player_initial_value']
        p_bust_odds = row['p_bust_odds']
        result = row['results']

        # Find the appropriate bucket for p_bust_odds
        bucket_index = int(p_bust_odds * 10)

        # Fixing issue where some values fell outside of expected bucket range
        if bucket_index < 0:
            bucket_index = 0
        elif bucket_index >= len(p_odds_buckets):
            bucket_index = len(p_odds_buckets) - 1

        p_odds_dict[bucket_index][0] += result
        p_odds_dict[bucket_index][1] += 1

    # Calculate the win ratio for each bucket
    win_ratios = [value[0] / value[1] if value[1] > 0 else 0 for value in p_odds_dict.values()]

    plt.bar(p_odds_buckets, win_ratios, color='purple')
    plt.ylabel('Win Ratio')
    plt.xlabel('Player Bust Odds Buckets')
    plt.title('Player Bust Odds vs. Win Ratio')
    plt.ylim(-.5, 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

win_vs_p_odds()

