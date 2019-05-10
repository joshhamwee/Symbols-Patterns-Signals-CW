from __future__ import print_function

import sys


def product(numbers):
    """Function to return the product of two numbers
    Params:
        numbers: List of two numbers to be multiplied
    Returns:
        product of two numbers
    """

    # Set base values
    finalValue = 1

    # Error catching
    if len(numbers) == 0:
        return 0


    # Iterate over list multiplying it to final value
    for x in numbers:
        x = int(x)
        finalValue *= x

    print(finalValue)


numbers = sys.argv[1:] # sys.argv contains the arguments passed to the program
product(numbers)
