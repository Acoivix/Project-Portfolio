from random import choice
import numpy as np


def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    # Checks to make sure out 3 digit number is 3 digits, and the first and last digit have a difference of at least 2
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("Number is not three digits buddy")
    first_digit, last_digit = int(step_1[0]), int(step_1[2])
    if abs(last_digit - first_digit) < 2:
        raise ValueError("Difference between last and first digit must be less than 2")

    # Checks to make sure number is properly reversed
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    reverse_number = step_1[::-1]
    if not reverse_number == step_2:
        raise ValueError("Number does not correctly equal reversed number")

    # Checks to make sure input is the correct positive difference
    step_3 = input("Enter the positive difference of these numbers: ")
    positive_difference = abs(int(step_1) - int(step_2))
    if not positive_difference == int(step_3):
        raise ValueError("Positive Difference is not Calculated Correctly")
    
    # Checks to make sure fourth number is reverse of third number
    step_4 = input("Enter the reverse of the previous result: ")
    reverse_number_2 = step_3[::-1]
    if not reverse_number_2 == step_4:
        raise ValueError("Reverse of number is not correct")
    
    #Prints our Magic Trick
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    
    # Raises Keyboard Interrup if control c is pushed, else just completes process
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print(f'Process interrupted at iteration {i}')
    else:
        print("Process Completed")
    
    # Return our walk regardless of whether process was stopped or completed
    return walk


class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        # Initialize our contents   
        self.contents = ''
        
        # Run our code through a for loop so that it keeps prompting our user for a valid filename
        while True:
            try:
                with open(filename, 'r') as myfile:         # opens our file
                    self.contents += myfile.read()
                    myfile.close()
                    break
            except (FileNotFoundError, TypeError, OSError) as e:
                filename = input('Please enter a valid file name:')          
                print(e)
        
        self.filename = filename    #Set our filename equal to our correctly inputted filename

        # Define all these variables for the str method
        self.total_chars = len(self.contents)
        self.alphabetical_chars = sum(i.isalpha() for i in self.contents)
        self.numerical_chars = sum(i.isdigit() for i in self.contents)
        self.whitespace_chars = sum(i.isspace() for i in self.contents)
        self.num_of_lines = len(self.contents.splitlines())

    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """

        #Check that our mode is w, x, or a
        if not mode in {"w", "x", "a"}:
            raise ValueError("The mode must be 'w', 'x', or 'a'")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include a
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        
        # Check that mode's valid
        self.check_mode(mode)

        # Writes data to our ourfile based on whether Caps or lower case is chosen
        if case == 'upper':
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.upper())
        elif case == 'lower':
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.lower())
        else:
            raise ValueError("Case must be 'lower' or 'upper'")       # Raises error if case inputted is not lower or upper


    def reverse(self, outfile, mode='w', unit='line'):
        """ Write the data to the outfile in reverse order. Include a
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        
        # Check that mode's valid
        self.check_mode(mode)

        # Obtain lines from our content
        lines = self.contents.splitlines()

        # Write our data to the outfile depending on unit given
        if unit == 'word':
            reversed_words = [' '.join(line.split()[::-1]) for line in lines]   
            reversed_words = '\n'.join(reversed_words)
            with open(outfile,mode) as myfile:
                myfile.write(reversed_words)
        elif unit == 'line':
            reversed_lines = lines[::-1]
            reversed_lines = '\n'.join(reversed_lines)
            with open(outfile, mode) as myfile:
                myfile.write(reversed_lines)
        else:
            raise ValueError("Unit must be 'word' or 'line'")

            
    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        
        # Check that mode's valid
        self.check_mode(mode)

        # Turn our data into lines, we're each line is split into indiviudal elements, essentially giving us a matrix
        lines = [line.split() for line in self.contents.splitlines()]
        
        # Transpose our "matrix"
        transposed_lines = np.transpose(lines)

        # Write our transposed matrix into the file
        with open(outfile, mode) as myfile:
            for i in transposed_lines:
                myfile.write(' '.join(i) + '\n')  
        
    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
        # Print our statistics calculated in the constructor
        return (f"Source file:              {self.filename}\n"
                f"Total characters:         {self.total_chars}\n"
                f"Alphabetic characters:    {self.alphabetical_chars}\n"
                f"Numerical characters:     {self.numerical_chars}\n"
                f"Whitespace characters:    {self.whitespace_chars}\n"
                f"Number of lines:          {self.num_of_lines}\n")