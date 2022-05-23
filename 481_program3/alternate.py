import sys
import numpy as np
import math
import random
import copy


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
        gameDict['board'] = result(gameDict['board'], action)
        gameDict['state'] += 1
        recursed = minVal(gameDict, turn)
        if v < recursed[0]:
            v = recursed[0]
            move = action
    return [v, move]


def minVal(gameDict, turn):
    v = sys.maxsize
    move = None
    if not isTerminal(gameDict):
        return [utility(gameDict['board']), move]
    for action in actions(gameDict['board'], gameDict['state']):
        gameDict['board'] = result(gameDict['board'], action)
        gameDict['state'] += 1
        recursed = maxVal(gameDict, turn)
        if v > recursed[0]:
            v = recursed[0]
            move = action
    return [v, move]


def result(board, action):
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
        graphDict[node] = copy.deepcopy(nodes2)
    return graphDict


def makeNet(numInputs, numHidden, numOutputs, numLayers):
    inpArr = list(range(1, numInputs + 1))
    outArr = list(range(1, numOutputs + 1))
    hidden = []
    for i in range(numLayers - 2):
        hidden.append(list(range(1, numHidden + 1)))
    realHidden = []
    for j in range(len(hidden)):
        realHidden.append(makeGraph(hidden[j], hidden[j]))
    net = {"inpWei": makeGraph(inpArr, hidden[0]),
           "outWei": makeGraph(hidden[-1], outArr), "hidWei": realHidden,
           "inpNode": inpArr, "outNode": outArr, "hidNode": hidden}
    return net


def initialize(weights):
    for key in weights.keys():
        for i in range(len(weights[key])):
            weights[key][i] = round(random.uniform(-2, 2), 3)
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
    netCopy = {}
    for key in network:
        netCopy[key] = copy.deepcopy(network[key])

    # initial layer to first hidden
    for key in netCopy['inpWei'].keys():
        for i in range(len(netCopy['inpWei'][key])):
            netCopy['hidNode'][0][i] = sigmoid(netCopy['inpWei'][key][i])

    # hidden to hidden
    for key in network['hidWei'][0].keys():
        for j in range(len(network['hidWei'][0][key])):
            netCopy['hidWei'][0][key][j] += netCopy['hidNode'][0][j]
    for key in network['hidWei'][0].keys():
        for j in range(len(network['hidWei'][0][key])):
            netCopy['hidNode'][1][j] = sigmoid(netCopy['hidWei'][0][key][j])

    for key in network['hidWei'][1].keys():
        for j in range(len(network['hidWei'][1][key])):
            netCopy['hidWei'][1][key][j] += netCopy['hidNode'][1][j]
    for key in network['hidWei'][1].keys():
        for j in range(len(network['hidWei'][1][key])):
            netCopy['hidNode'][2][j] = sigmoid(netCopy['hidWei'][1][key][j])

    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['hidWei'][2][key][j] += netCopy['hidNode'][2][j]
    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['hidNode'][3][j] = sigmoid(netCopy['hidWei'][2][key][j])

    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['hidWei'][3][key][j] += netCopy['hidNode'][3][j]
    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['outWei'][key][j] = sigmoid(netCopy['hidWei'][3][key][j])

    # hidden final to output
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['hidWei'][3][key][j] += netCopy['hidNode'][3][j]
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['outNode'][j] = sigmoid(netCopy['hidWei'][3][key][j])

    return netCopy


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
    netCopy = {}
    for key in network:
        netCopy[key] = copy.deepcopy(network[key])

    # initial layer to first hidden
    for key in netCopy['inpWei'].keys():
        for i in range(len(netCopy['inpWei'][key])):
            netCopy['hidNode'][0][i] = relu(netCopy['inpWei'][key][i])

    # hidden to hidden
    for key in network['hidWei'][0].keys():
        for j in range(len(network['hidWei'][0][key])):
            netCopy['hidWei'][0][key][j] += netCopy['hidNode'][0][j]
    for key in network['hidWei'][0].keys():
        for j in range(len(network['hidWei'][0][key])):
            netCopy['hidNode'][1][j] = relu(netCopy['hidWei'][0][key][j])

    for key in network['hidWei'][1].keys():
        for j in range(len(network['hidWei'][1][key])):
            netCopy['hidWei'][1][key][j] += netCopy['hidNode'][1][j]
    for key in network['hidWei'][1].keys():
        for j in range(len(network['hidWei'][1][key])):
            netCopy['hidNode'][2][j] = relu(netCopy['hidWei'][1][key][j])

    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['hidWei'][2][key][j] += netCopy['hidNode'][2][j]
    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['hidNode'][3][j] = relu(netCopy['hidWei'][2][key][j])

    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['hidWei'][3][key][j] += netCopy['hidNode'][3][j]
    for key in network['hidWei'][2].keys():
        for j in range(len(network['hidWei'][2][key])):
            netCopy['outWei'][key][j] = relu(netCopy['hidWei'][3][key][j])

    # hidden final to output
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['hidWei'][3][key][j] += netCopy['hidNode'][3][j]
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            netCopy['outNode'][j] = sigmoid(netCopy['hidWei'][3][key][j])

    return netCopy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return max(x, 0)


def backPropSig(network, gameDict):
    val = minmax(gameDict)
    if val[1] == None:
        return network
    val = val[-1].split(' ')[-1].split('-')[-1]
    indList = ['c', 'b', 'a']
    checkList = [0] * 9
    index = indList.index(val[0]) * 3 + int(val[1]) - 1
    checkList[index] = 1
    errorList = []
    weightList = []

    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][3].keys():
            weightSum += network['hidWei'][3][key][i]
            weightList.append(weightSum)
    for i in range(9):
        error = updateWeightsSig(network['outNode'][i], checkList[i],
                                 weightList[i])
        errorList.append(error)
    for key in network['outWei'].keys():
        for i in range(9):
            network['outWei'][key][i] -= errorList[i]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][2][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][3][j] - network['hidWei'][3][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsSig(network['hidNode'][3][i], checkList[i],
                                 weightList[i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            network['hidNode'][3][j] -= errorList[j]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][1][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][2][j] - network['hidWei'][2][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsSig(network['hidNode'][2][i], checkList[i],
                                 weightList[i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][2][key])):
            network['hidNode'][2][j] -= errorList[j]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][0][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][1][j] - network['hidWei'][1][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsSig(network['hidNode'][1][i], checkList[i],
                                 weightList[i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][2][key])):
            network['hidNode'][1][j] -= errorList[j]

    return network


def backPropRelu(network, gameDict):
    val = minmax(gameDict)
    if val[1] == None:
        return network

    val = val[-1].split(' ')[-1].split('-')[-1]
    indList = ['c', 'b', 'a']
    checkList = [0] * 9
    index = indList.index(val[0]) * 3 + int(val[1]) - 1
    checkList[index] = 1
    errorList = []
    weightList = []

    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][3].keys():
            weightSum += network['hidWei'][3][key][i]
            weightList.append(weightSum)
    for i in range(9):
        error = updateWeightsRelu(network['outNode'][i])
        errorList.append(error)
    for key in network['outWei'].keys():
        for i in range(9):
            network['outWei'][key][i] -= errorList[i]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][2][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][3][j] - network['hidWei'][3][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsRelu(network['hidNode'][3][i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][3][key])):
            network['hidNode'][3][j] -= errorList[j]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][1][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][2][j] - network['hidWei'][2][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsRelu(network['hidNode'][2][i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][2][key])):
            network['hidNode'][2][j] -= errorList[j]

    errorList = []
    checkList = []
    for i in range(9):
        weightSum = 0
        for key in network['hidWei'][2].keys():
            weightSum += network['hidWei'][0][key][i]
            weightList.append(weightSum)
    for key in network['hidWei'][3].keys():
        check = 0
        for j in range(len(network['hidWei'][3][key])):
            check += network['hidNode'][1][j] - network['hidWei'][1][key][j]
        checkList.append(check)
    for i in range(9):
        error = updateWeightsRelu(network['hidNode'][1][i])
        errorList.append(error)
    for key in network['hidWei'][3].keys():
        for j in range(len(network['hidWei'][2][key])):
            network['hidNode'][1][j] -= errorList[j]

    return network


def updateWeightsSig(actual, expected, weighted_sum):
    weighted_sum = sigmoid(weighted_sum)
    return (2 * (actual - expected) * (
                sigmoid(weighted_sum) * (1 - sigmoid(weighted_sum))))


def updateWeightsRelu(actual):
    if actual < 0:
        return 0
    return 1


def main():
    # -1 = white; 1 = black; 0 = blank space
    gameDict = {'board': [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                'state': 0}

    classifiedSig = makeNet(10, 9, 9, 6)
    classifiedRelu = makeNet(10, 9, 9, 6)

    for i in range(1000):
        classifiedSig = classifySig(classifiedSig, gameDict)
        classifiedRelu = classifyRelu(classifiedRelu, gameDict)

        classifiedSig = backPropSig(classifiedSig, gameDict)
        classifiedRelu = backPropRelu(classifiedRelu, gameDict)

    print(classifiedSig['outNode'])
    print(classifiedRelu['outNode'])


if __name__ == '__main__':
    main()
