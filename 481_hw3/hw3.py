"""
hw3.py
James Villemarette
CISC 481, Homework 3
Due May 17, 2022
"""

# PART 1


class Hexapawn:
    """
    Implement a Formalization of Hexapawn based upon the definition given in
    Section 5.1.1 of the book.
    """

    def __init__(self):
        pass

    def to_move(self, s):
        """
        The player whose turn it is to move in state.
        :param s: State s.
        :return: Which player is moving.
        """
        pass

    def actions(self, s):
        """
        The set of legal moves in state s
        :param s: State s.
        :return: All legal moves.
        """
        pass

    def result(self, s, a):
        """
        The transition model, which defines the state resulting from taking
        action a in state s.
        :param s: State s.
        :param a: Action a.
        :return: The resulting state.
        """
        pass

    def is_terminal(self, s):
        """
        A terminal test, which is true when the game is over and false
        otherwise. States where the game has ended are called terminal states.
        :param s: State s.
        :return: True if the game is over, False otherwise.
        """
        pass

    def utility(self, s, p):
        """
        A utility function (also called an objective function or payoff
        function), which defines the final numeric value to player when the game
        ends in terminal state. In chess, the outcome is a win, loss, or draw,
        with values, 0, or 1/2.

        Some games have a wider range of possible outcomes—for example, the
        payoffs in backgammon range from 0 to .
        :return:
        """
        pass


# PART 2

def minimax_search():
    """
    Implement a minimax search that builds up a policy table for a game that has
    been formalized as you have for Hexapawn in Part 1. For each state, the
    policy should include the value of the game as well as every action that
    achieves that value.
    :return:
    """
    pass

# PART 3


