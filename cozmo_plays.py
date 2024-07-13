# This code is adapted from Ali Nasser's cozmo qr and network code
# I added my own model decision making into in and replaced the old decision making along with some other minor tweaks

# Import required libraries and modules
import cozmo
import socket
from socket import error as socket_error
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import cv2
import qrcode

# Adjust number of decks here
num_decks = 4

# Define Cozmo's name
COZMO_NAME = "isaac"

# Table number
TABLE_NUM = 1

# Position at table
POSITION = 6

# Dictionary from simulator
card_count = {'A':0, 2:0, 3:0, 4:0, 5:0, 6:0,7:0, 8:0, 9:0, 10:0, 'J':0, 'Q':0, 'K':0}
poker_ranks_to_nums = {
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5,
    'Six': 6,
    'Seven': 7,
    'Eight': 8,
    'Nine': 9,
    'Ten': 10,
    'Jack': 10,
    'Queen': 10,
    'King': 10,
    'Ace': 11
}
card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'] 

# Find total returns the total int total of a card or hand
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
    
# Function to calculate the bust odds
def bust_chance(hand_total, shoe):
    remaining_cards = 0
    bust_cards = 0
    per_rank = num_decks * 4
    # range(10) gives 0-9, i want 1-10
    for i in card_types:
        # If the card would make us bust add the total number of that card remaining in the deck to bust_cards
        if hand_total + find_total(i) + 1 > 21:
            bust_cards += (per_rank - shoe[i])
            remaining_cards += (per_rank - shoe[i])
        # Whether or not we bust with the card, add it to the remaining cards to use as the divisor
        else:
            remaining_cards += (per_rank - shoe[i])
    return bust_cards/remaining_cards

# Function to convert card face values to their numerical equivalents
def card_value(value):
    if value in ["Jack", "Queen", "King"]:
        return 10
    elif value == "Ace":
        return 11
    else:
        return int(value)
    
# Get true count function
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
    
# Model_decision function loads the ML models and takes the information from cozmo to determine if he should hit or stay
#first, load the model
new_model = tf.keras.models.load_model('basic_model.keras')

def model_decision(model, player_value, dealer_card, p_bust_odds, card_count):
    
    d = {'dealer_card':[dealer_card], 'init_hand':[player_value], 'p_bust_odds':[p_bust_odds], 'hit':[1], 'card_count':[card_count]} 
    
    input_df = pd.DataFrame(data=d)
    
    #run the data through the model
    prediction = model.predict(input_df)
    
    #at this stage, we want to know what the model predicted
    print(prediction)
    
    #now we can fine-tune our decision (hit or stay) based on the model's prediction
    if prediction > 0.4:
        return 1
    else:
        #we can be conservative here and just stay
        return 0


def lookForCards( robot: cozmo.robot.Robot, hand):
    # Turn on camera
    robot.camera.image_stream_enabled = True
    # robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
 
    # Instantiate QR code detector
    detector = cv2.QRCodeDetector()
 
    # Look for codes
    while True:
        # Get current image
        image = robot.world.latest_image
 
        if image is not None:
            # Check for qr codes
            code_detected, decoded_string, _, _ = detector.detectAndDecodeMulti(
                np.array(image.raw_image)
            )

            # If a qr code was detected, then process it
            if code_detected and decoded_string[0] != "":
                card_info = decoded_string[0].strip("()").split(';')
                if card_info[0] not in hand:
                    time.sleep(1)
                    return card_info

# Main program to be executed on the Cozmo robot
def cozmo_program(robot: cozmo.robot.Robot):
    # Enable the robot's camera stream
    robot.camera.image_stream_enabled = True
    total_value = 0  # Total value of detected cards
    cards_detected = []  # List to keep track of detected cards
    dealer_card = 'Ten;Clubs'
    bytedata = 0
    skip = False

    # Initialize socket connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket_error as msg:
        robot.say_text("socket failed" + msg).wait_for_completed()
    ip = "10.0.1.10"
    port = 5000
    
    # Attempt to connect to the specified IP and port
    try:
        s.connect((ip, port))
    except socket_error as msg:
        robot.say_text("socket failed to bind").wait_for_completed()

    robot.say_text("ready").wait_for_completed()

    start_message = f'{TABLE_NUM};{COZMO_NAME};{POSITION}'
    s.sendall(start_message.encode('utf-8'))

    cont = True
    while cont:
        
        card = None
        time.sleep(1)
        # Fetch the latest image from the robot's camera
        image = robot.world.latest_image.raw_image
        image = image.convert('L')
        decoded = decode(image, symbols=[ZBarSymbol.QRCODE])  # Decode QR codes from the image

        if len(decoded) > 0:
            codeData = decoded[0]
            myData = codeData.data
            myString = myData.decode('ASCII')
            print(myString)
            card = myString
        else:
            print('I could not decode the data')

        if card and card not in cards_detected:
            cards_detected.append(card)  # Add card to detected list
            value, suit = card.split(";")  # Extract card value and suit
            card_val = poker_ranks_to_nums[value]  # Convert card value to numerical equivalent
            total_value += card_val  # Update total hand value

            # Send the card over the network
            message = f'{TABLE_NUM};{COZMO_NAME};{len(cards_detected)};{card[0]}'
            s.sendall(message.encode('utf-8'))

            # Handle Ace values when total exceeds 21
            if value == "Ace" and total_value > 21:
                total_value -= 10

            #robot.say_text(f"{value} of {suit}, hand value is {total_value}").wait_for_completed()
            print(f"{value} of {suit}, hand value is {total_value}")

            # After detecting at least two cards
            if len(cards_detected) >= 2:
                action = ['stay', 'hit']

                # Dealer card should be taken from network this is just for testing
                dealer_card = dealer_card.split(';')
                dealer_rank = poker_ranks_to_nums[dealer_card[0]]

                # Use the current card count dict and num_deck to get the current count
                set_count = get_count(card_count, num_decks)

                # Use the card count dict to get the player bust odd
                b_odds = bust_chance(total_value, card_count)

                decision = action[model_decision(new_model, total_value, dealer_rank, b_odds, set_count)]

                # Decision to HIT or STAY based on total hand value
                if decision == 'hit':
                    robot.say_text("HIT").wait_for_completed()
                    robot.set_lift_height(1).wait_for_completed()  # Move lift up
                    robot.set_lift_height(0).wait_for_completed()  # Move lift down
                else:
                    robot.say_text("STAY").wait_for_completed()
                    #robot.turn_in_place(cozmo.util.degrees(360)).wait_for_completed()  # Spin around
                    # Reset the total value and hand but leave everything else
                    stay_message = f'{TABLE_NUM};{COZMO_NAME};-1;-1'
                    s.sendall(stay_message.encode('utf-8'))
                    total_value = 0
                    cards_detected = []
                    dealer_card = ''

                    #message = f"{COZMO_NAME};{''.join(cards_detected[0])};{''.join(cards_detected[1])}"
                    #print(message)
                    #s.sendall(message.encode('utf-8'))
                    #break  # Exit the loop


    s.close()  # Close the socket connection

# Start the Cozmo program
cozmo.run_program(cozmo_program,True,force_viewer_on_top=True)