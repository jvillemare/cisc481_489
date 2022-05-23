import sys
import numpy as np
import math
import random


def toMove(state):
    """
    Determine which player's turn it is
    :param state: The state of the game.
    :return: A string of which player whose turn it is, 'player1' or 'player2'.
    """
    # whos moving?
    if state % 2:
        return 'player2'
    return 'player1'


def actions(board, state):
    """
    Generate a list of actions based of the current board and state.
    :param board: The current board.
    :param state: The current state.
    :return: List of legal actions.
    """
    # what moves can you make?
    legals = []
    if state % 2:
        # black to move
        # front moves ::
        # black innitial row move fronts
        if board[0][0] == 1:
            if not (board[1][0]):
                legals.append("a3-a2")
                if not (board[2][0]):
                    legals.append("a3-a1")
        if board[0][1] == 1:
            if not (board[1][1]):
                legals.append("b3-b2")
                if not (board[2][1]):
                    legals.append("b3-b1")
        if board[0][2] == 1:
            if not (board[1][2]):
                legals.append("c3-c2")
                if not (board[2][2]):
                    legals.append("c3-c1")

        # black middle row move fronts
        if board[1][0] == 1:
            if not (board[2][0]):
                legals.append("a2-a1")
        if board[1][1] == 1:
            if not (board[2][1]):
                legals.append("b2-b1")
        if board[1][2] == 1:
            if not (board[2][2]):
                legals.append("c2-c1")

        # black takes initial
        if board[0][0] == 1:
            if board[1][1] == -1:
                legals.append("takes a3-b2")
        if board[0][1] == 1:
            if board[1][0] == -1:
                legals.append("takes b3-a2")
            if board[1][2] == -1:
                legals.append("takes b3-c2")
        if board[0][2] == 1:
            if board[1][1] == -1:
                legals.append("takes c3-b2")

        # black takes middle
        if board[1][0] == 1:
            if board[2][1] == -1:
                legals.append("takes b1-a2")
        if board[1][1] == 1:
            if board[2][0] == -1:
                legals.append("takes b2-a1")
            if board[2][2] == -1:
                legals.append("takes b2-a3")
        if board[1][2] == 1:
            if board[2][1] == -1:
                legals.append("takes b3-a2")

    else:
        # white to move
        # front moves ::
        # white innitial row move fronts
        if board[2][0] == -1:
            if not (board[1][0]):
                legals.append("a1-a2")
                if not (board[0][0]):
                    legals.append("a1-a3")
        if board[2][1] == -1:
            if not (board[1][1]):
                legals.append("b1-b2")
                if not (board[0][1]):
                    legals.append("b1-b3")
        if board[2][2] == -1:
            if not (board[1][2]):
                legals.append("c1-c2")
                if not (board[0][2]):
                    legals.append("c1-c3")

        # white middle row move fornts
        if board[1][0] == -1:
            if not (board[0][0]):
                legals.append("a2-a3")
        if board[1][1] == -1:
            if not (board[0][1]):
                legals.append("b2-b3")
        if board[1][2] == -1:
            if not (board[0][2]):
                legals.append("c2-c3")

        # white takes initial
        if board[2][0] == -1:
            if board[1][1] == 1:
                legals.append("takes a1-b2")
        if board[2][1] == -1:
            if board[1][0] == 1:
                legals.append("takes a2-b1")
            if board[1][2] == 1:
                legals.append("takes a2-b3")
        if board[2][2] == -1:
            if board[1][1] == 1:
                legals.append("takes a3-b2")

        # white takes middle
        if board[1][0] == -1:
            if board[0][1] == 1:
                legals.append("takes b1-c2")
        if board[1][1] == -1:
            if board[0][0] == 1:
                legals.append("takes b2-c1")
            if board[0][2] == 1:
                legals.append("takes b2-c3")
        if board[1][2] == -1:
            if board[0][1] == 1:
                legals.append("takes b3-c2")
    return legals


def isTerminal(gameDict):
    """
    Is the game over at this state?
    :param gameDict:
    :return:
    """
    # is the game over?
    acts = actions(gameDict['board'], gameDict['state'])
    return bool(len(acts))


def utility(board):
    """
    The
    :param board:
    :return:
    """
    # how did the game end? win (1), loss (0) or draw (1/2)?
    if 1 in board[2]:
        return 1
    if -1 in board[0]:
        return -1
    return 1 / 2


def minmax(gameDict):
    # minmax search
    turn = toMove(gameDict['state'])
    util_move = maxVal(gameDict, turn)
    return util_move


def maxVal(gameDict, turn):
    v = -sys.maxsize - 1
    move = None
    if not isTerminal(gameDict):
        return [utility(gameDict['board']), move]
    for action in actions(gameDict['board'], gameDict['state']):
        gameDict['board'] = doAction(gameDict['board'], action)
        gameDict['state'] += 1
        recursed = minVal(gameDict, turn)
        if v < recursed[0]:
            v = recursed[0]
            move = action
            print(v)
            print(action)
    return [v, move]


def minVal(gameDict, turn):
    v = sys.maxsize
    move = None
    if not isTerminal(gameDict):
        return [utility(gameDict['board']), move]
    for action in actions(gameDict['board'], gameDict['state']):
        gameDict['board'] = doAction(gameDict['board'], action)
        gameDict['state'] += 1
        recursed = maxVal(gameDict, turn)
        if v > recursed[0]:
            v = recursed[0]
            print(v)
            move = action
            print(action)
    return [v, move]


def doAction(board, action):
    action = action.split(' ')[-1].split('-')
    rows = ['a', 'b', 'c']
    cols = ['3', '2', '1']
    curRow = int(rows.index(action[0][0]))
    curCol = int(cols.index(action[0][1]))
    modRow = int(rows.index(action[1][0]))
    modCol = int(cols.index(action[1][1]))
    inVal = board[curCol][curRow]
    board[curCol][curRow] = 0
    board[modCol][modRow] = inVal
    return board


def makeGraph(nodes1, nodes2):
    graphDict = {}
    for node in nodes1:
        graphDict[node] = nodes2
    return graphDict


def makeNet(numInputs, numHidden, numOutputs, numLayers):
    inpArr = list(range(1, numInputs + 1))
    outArr = list(range(1, numOutputs + 1))
    hidden = []
    for i in range(numLayers - 2):
        hidden.append(list(range(1, numHidden + 1)))
    realHidden = []
    for h in hidden:
        realHidden.append(makeGraph(hidden[0], hidden[0]))
    net = {"inpWei": makeGraph(inpArr, hidden[0]),
           "outWei": makeGraph(hidden[-1], outArr), "hidWei": realHidden,
           "inpNode": inpArr, "outNode": outArr, "hidNode": hidden}
    return net


def initialize(weights):
    for key in weights.keys():
        for i in range(len(weights[key])):
            weights[key][i] = round(random.uniform(-1, 1), 3)
    return weights


def classifySig(network, gameDict):
    network['inpNode'][0] = gameDict['state']
    for i in range(9):
        network['inpNode'][i + 1] = gameDict['board'][math.trunc(i / 3)][i % 3]
    network['inpWei'] = initialize(network['inpWei'])
    for i in range(len(network['hidWei'])):
        network['hidWei'][i] = initialize(network['hidWei'][i])
    return runClassifySig(network)


def runClassifySig(network):
    for key in network['inpWei'].keys():
        for i in range(len(network['inpWei'][key])):
            network['inpWei'][key][i] += network['inpNode'][int(key) - 1]
    for key in network['inpWei'].keys():
        for i in range(len(network['inpWei'][key])):
            network['inpWei'][key][i] = 1 / (
                        1 + math.exp(-network['inpWei'][key][i]))
    return network


def classifyRelu(network, gameDict):
    network['inpNode'][0] = gameDict['state']
    for i in range(9):
        network['inpNode'][i + 1] = gameDict['board'][math.trunc(i / 3)][i % 3]
    network['inpWei'] = initialize(network['inpWei'])
    for i in range(len(network['hidWei'])):
        network['hidWei'][i] = initialize(network['hidWei'][i])
    return runClassifyRelu(network)


def runClassifyRelu(network):
    for key in network['inpWei'].keys():
        for i in range(len(network['inpWei'][key])):
            network['inpWei'][key][i] += network['inpNode'][int(key) - 1]
    for key in network['inpWei'].keys():
        for i in range(len(network['inpWei'][key])):
            network['inpWei'][key][i] = max(network['inpWei'][key][i], 0)
    return network


def main():
    # -1 = white; 1 = black; 0 = blank space
    gameDict = {'board': [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                'state': 4}

    testDict = {'board': [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                'state': 0}

    netDict = makeNet(10, 15, 9, 6)
    classifiedSig = classifySig(netDict, gameDict)
    print(classifiedSig['inpWei'])
    classifiedRelu = classifyRelu(netDict, gameDict)
    print(classifiedRelu['inpWei'])

    # print(initialize(netDict['inpWei']))


if __name__ == '__main__':
    main()