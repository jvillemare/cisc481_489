import sys
import numpy as np


def toMove(state):
    # whos moving?
    if state % 2:
        return 'player2'
    return 'player1'


def actions(board, state):
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


def result(board, state):
    # what happens if you make that move?
    return


def isTerminal(gameDict):
    # is the game over?
    acts = actions(gameDict['board'], gameDict['state'])
    return bool(len(acts))


def utility(board):
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
    net = {"inputs": makeGraph(inpArr, hidden[0]),
           "outputs": makeGraph(hidden[-1], outArr), "hidden": realHidden}
    return net


def main():
    # -1 = white; 1 = black; 0 = blank space
    gameDict = {'board': [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                'state': 0}

    testDict = {'board': [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                'state': 0}

    print(makeNet(4, 7, 2, 5))


if __name__ == '__main__':
    main()
