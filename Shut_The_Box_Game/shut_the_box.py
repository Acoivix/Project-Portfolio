from itertools import combinations
import sys
import random


def isvalid(roll, remaining):
    """Checks to see whether or not a roll is valid. That is, check if there
    exists a combination of the entries of 'remaining' that sum up to 'roll'.
    Args:
        roll (int): The roll value to be validated.
        remaining (list): A list of available numbers to choose from.
    
    Returns:
        bool: True if a valid combination exists, False otherwise.
    """
    if roll not in range(1, 13):
        return False
    for i in range(1, len(remaining)+1):
        if any([sum(combo) == roll for combo in combinations(remaining, i)]):
            return True
    return False


def parse_input(player_input, remaining):
    """Convert a string of numbers into a list of unique integers, if possible.
    Then check that each of those integers is an entry in the other list.
    Args:
        player_input (str): A string of numbers entered by the player.
        remaining (list): A list of available numbers to choose from.
    
    Returns:
        list: A list of valid choices, or an empty list if input is invalid.
    """
    try:
        choices = [int(i) for i in player_input.split()]
        if len(set(choices)) != len(choices):
            raise ValueError
        if any([number not in remaining for number in choices]):
            raise ValueError
        return choices
    except ValueError:
        return []


def shut_the_box(player1, player2):
    """Play a single game of shut the box between 2 players"""
    """In order to start the game, input it into the terminal as python (file) (player1) (player2)"""
    """Rules:                                     """
    """Player 1 Begins to play the game as normal trying to remove each number from our list of 9 numbers"""
    """Whenever Player 1 does not have the ability to get rid of any more of the numbers, the turn goes to Player 2 """
    """Player 2 then is trying to add every number back to list of 9 numbers"""
    """If Player 2 cannot make any combinations that return a given number back, it goes back to Player 1"""
    """The game ends when either Player 1 can remove all the numbers, or Player 2 can return all the numbers to the original list """
    """When entering numbers enter them 2 2  or 3 1   or 5 6  or 4 2 3 . No + signs"""
    
    
    if len(args) == 3: #start game only if we have the right amount of argvs
        numbers_left = list(range(1,10)) #initialize our numbers left for player 1
        player2_numbers_left = [] #the numbers that player1 eliminates will be added back here so that player2 has the chance to add them back

        turn = 1  #initialize that it is player1's turn

        while sum(numbers_left) > 0 or len(player2_numbers_left)> 0:   
            if turn == 1:    #While its player1's turn
                print(f"Your turn: {player1}")
                while True:         #Continues to give player1 numbers as long as they can eliminate more
                    print(f"Numbers left for {player1}: {numbers_left}")   #Prints our numbers left

                    if sum(numbers_left) <= 6:                 #Get our new dice roll and print it
                        dice_roll = random.randint(1, 6)
                    else:
                        dice_roll = random.randint(1, 6) + random.randint(1, 6)
                    print(f"Roll: {dice_roll}")
    
                    if not isvalid(dice_roll, numbers_left):      #Changes to player 2 if there is no combinations
                        print(f"{player1}, no valid combinations. Turn passes to {player2}")
                        print()
                        turn = -1
                        break
                    
                
                    while True:      # Lets user choose numbers to eliminate, and checks to make sure the numbers are valid                                
                        nums = input(f"{player1}'s Numbers to eliminate: ")
                        chosen_numbers = parse_input(nums, numbers_left)
                        if chosen_numbers is None or isvalid(dice_roll, chosen_numbers):
                            for number in chosen_numbers:
                                numbers_left.remove(number)    #Remove the numbers user has chosen as long as they're valid
                                player2_numbers_left.append(number) #Adds the number to player 2's list of numbers they need to add back to win
                                print()        
                            player2_numbers_left.sort()
                            numbers_left.sort()
                            break
                        else:
                            print("Invalid Input, Try again!\n")
            
                
                    if len(numbers_left) == 0:        #If player1 wins
                        print(f"Congratulations {player1}!! You are victorious!")
                        print(f"{player2}, you have lost :(")
                        break

            elif turn == -1:     #plays the game but for player 2
                print(f"Your turn, {player2}") 
                while True:
                    print(f"Numbers left for {player2} to add: {player2_numbers_left}")
                    
                    dice_roll = random.randint(1, 6) + random.randint(1, 6)
                    print(f"Roll: {dice_roll}")

                    if not isvalid(dice_roll, player2_numbers_left):
                        print(f"{player2}, no valid combinations. Turn passes to {player1}.")
                        print()
                        turn = 1
                        break  # End Player 2's turn if no valid combinations exist

                    while True:
                        nums = input(f"{player2}, choose numbers to add: ")
                        chosen_numbers = parse_input(nums, player2_numbers_left)
                        if chosen_numbers and isvalid(dice_roll, chosen_numbers):
                            for number in chosen_numbers:
                                player2_numbers_left.remove(number)   #removes from numbers to add for player2 if valid
                                numbers_left.append(number)
                            player2_numbers_left.sort()
                            numbers_left.sort()
                            break
                        else:
                            print("Invalid input, try again.")

                if sum(player2_numbers_left) == 0:   #Player 2 Win condition
                    print(f"Congratulations {player2}! You are Victorious!")
                    print(f"{player1}, you have lost :(")
                    return  


if __name__ == "__main__":
    args = sys.argv 
    player1 = args[1]
    player2 = args[2]
    shut_the_box(player1, player2)