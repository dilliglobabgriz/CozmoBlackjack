import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns

def hands_heat_map(new_model):
    def get_predicted_values(model, player_value, dealer_card):
        # Assume that the player always stays (hit = 0)
        # You can change this if you want to consider hit as well
        num_hits = 0
        count = 0
        prediction = model_decision(model, dealer_card, player_value, card_count, hit_count)
        return prediction

    # Create a table of predicted values
    def create_mega_table():
        player_hands = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        dealer_init = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        data = []
        for player_value in player_hands:
            for dealer_card in dealer_init:
                prediction = get_predicted_values(new_model, player_value, dealer_card)
                data.append([player_value, dealer_card, prediction])
        
        # Create a Pandas DataFrame
        table = pd.DataFrame(data, columns=['Player Hand', 'Dealer Initial Card', 'Predicted Value'])
        return table

    # Call the function to create the table
    predicted_values_table = create_mega_table()

    # Ensure the data is in a numeric format
    predicted_values_table['Predicted Value'] = predicted_values_table['Predicted Value'].astype(float)

    # Reshape the data for heatmap
    heatmap_data = predicted_values_table.pivot(index='Player Hand', columns='Dealer Initial Card', values='Predicted Value')

    # Reverse the y-axis labels
    heatmap_data = heatmap_data.iloc[::-1]

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Blackjack Predicted Values Heatmap Count: 0 Num_Hits: 0')
    plt.xlabel('Dealer Initial Card')
    plt.ylabel('Player Hand value')
    plt.show()

def count_hits_heat_map(new_model):
    def get_predicted_values(model, card_count, hit_count):
        # Assume that the player always stays (hit = 0)
        # You can change this if you want to consider hit as well
        dealer_value = 5
        player_value = 10
        prediction = model_decision(model, dealer_value, player_value, card_count, hit_count)
        return prediction

    # Create a table of predicted values
    def create_mega_table():
        counts = []
        for i in range(-10, 10):
            counts.append(i)
        hits = [0, 1, 2, 3, 4, 5]
        
        data = []
        for count in counts:
            for hit in hits:
                prediction = get_predicted_values(new_model, count, hit)
                data.append([count, hit, prediction])
        
        # Create a Pandas DataFrame
        table = pd.DataFrame(data, columns=['Count', 'Num Hits', 'Predicted Value'])
        return table

    # Call the function to create the table
    predicted_values_table = create_mega_table()

    # Ensure the data is in a numeric format
    predicted_values_table['Predicted Value'] = predicted_values_table['Predicted Value'].astype(float)

    # Reshape the data for heatmap
    heatmap_data = predicted_values_table.pivot(index='Count', columns='Hit Count', values='Predicted Value')

    # Reverse the y-axis labels
    heatmap_data = heatmap_data.iloc[::-1]

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Blackjack Predicted Values Heatmap Dealer: 5 Player: 10')
    plt.xlabel('Hit Count')
    plt.ylabel('Count')
    plt.show()


def model_decision(model, dealer_card, player_value, card_count, hit_count):
    
    d = {'dealer_card':[dealer_card], 'player_initial_value':[player_value], 'hit':[1], 'count':[card_count], 'num_hits': [hit_count]} 

    input_df = pd.DataFrame(data=d)
    
    #run the data through the model
    prediction = model.predict(input_df)
    
    return prediction

def get_model_stats(algorithm):

    new_model = tf.keras.models.load_model('basic_model.keras')

    final_df = pd.read_csv(f'blackjack/sims_csvs/blackjackdata{algorithm}20k.csv')


    #first, determine the features to include.  i will include the dealer card that is showing, the 
    #cards that the player has been dealt, and whether the player hit or not.  I will not include card
    #counting, or an awareness of the number of players at the table or the number of decks of cards
    #in the shoe.  That might be something that you include in your model.

    feature_list = ['dealer_card','player_initial_value','hit', 'card_count', 'hit_count']

    #i need to address the problem of the dealer card being numberical and string data.
    #i want the dealer card to be numerical in nature, so I'll use the replace method
    #and do this in place.  If you wonder what that means, try not using that attribute
    #or setting it to be False
    final_df['dealer_card'].replace({'A':11, 'J':10, 'Q':10, 'K':10}, inplace=True)

    #to build the model, i need to extract the information in my feature list (omitting 
    #unnecessary features as well as the label, or the attribute that I want my model to predict
    #make sure that the data is in a form that con be converted to a tensor...

    #X_df = final_df[feature_list]
    X_df = np.array(final_df[feature_list]).astype(np.float32)

    #given the dealer card, the player's hand, and their action (hit or stay) was that the correct choice?
    #for my model predition, i will default to my input being the dealer card, the player's hand, and they hit
    #the question will be was it the correct decision.  if so, then cozmo should hit.  if not, then cozmo should stay.
    #again, your reasoning might be different.  again, make sure that your data is in a form that can
    #be converted to a tensor

    # y_df = final_df['outcome']
    y_df = np.array(final_df['outcome']).astype(np.float32).reshape(-1,1)

    #next, break up the data into trining data and testing data...20% of the data will be used to evaluate
    #the model, and 80% of the data will be used to train the model.  You can change these parameters
    #to explore the impact.  we are using the train_test_split method we imported.
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.2)


    #make some predictions based on the test data that we reserved
    pred_Y_test = new_model.predict(X_test)
    #also get the actual results so we can compare
    actuals = y_test

    fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_test)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(fpr, tpr, label = ('ROC AUC = %0.3f' % roc_auc))

    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)
    plt.tight_layout()
    plt.savefig(fname='roc_curve_blackjack', dpi=150)
    plt.show()

    # Assuming pred_Y_test contains continuous probabilities
    thresholds = []
    for i in range(35, 60):
        thresholds.append(i / 100)

    accuracy_per_threshold = []  # Replace with your actual accuracy data
    for thres in thresholds:
        binary_pred_Y_test = (pred_Y_test >= thres).astype(int)
        # Assuming actuals are binary (0 or 1)

        # Assuming actuals and binary_pred_Y_test are set correctly as mentioned earlier
        accuracy = metrics.accuracy_score(actuals, binary_pred_Y_test) * 100
        accuracy_per_threshold.append(accuracy)

    max_accuracy = max(accuracy_per_threshold)
    min_accuracy = min(accuracy_per_threshold)

    # Create an array of x values for the line plot
    x = np.arange(len(thresholds))
    # Create a line plot
    plt.plot(x, accuracy_per_threshold, marker='o', linestyle='-', color='b')

    # Set the x-axis labels
    plt.xticks(x, thresholds)

    # Set axis labels and chart title
    plt.xlabel('Threshold')
    plt.ylabel('Percent')
    plt.ylim(min(accuracy_per_threshold) - 5, max(accuracy_per_threshold) + 5)
    plt.title('Percent accurate')

    # Add the maximum and minimum values as labels
    plt.text(x[accuracy_per_threshold.index(max_accuracy)], max_accuracy + 1, f'{max_accuracy:.2f}%', ha='center', va='bottom')
    plt.text(x[accuracy_per_threshold.index(min_accuracy)], min_accuracy - 1, f'{min_accuracy:.2f}%', ha='center', va='top')

    plt.scatter(x[accuracy_per_threshold.index(max_accuracy)], max_accuracy, color='red', marker='*', s=100, label=f'Max: {max_accuracy:.2f}%')


    plt.show()
    max_index = accuracy_per_threshold.index(max(accuracy_per_threshold))
    binary_pred_Y_test = (pred_Y_test >= thresholds[max_index]).astype(int)
    conf_matrix = confusion_matrix(actuals, binary_pred_Y_test)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    hands_heat_map(new_model)
    count_hits_heat_map(new_model)


    

choice = 'count_smart'
get_model_stats(choice)
