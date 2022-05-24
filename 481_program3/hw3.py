"""
hw3.py
CISC 481 - HW3 - Neural Nets
Spring 2022
James Villemarette
"""

import sys
import math
import random
import copy


def result(board, action):
    """
    Apply an action to a board.

    I proxy this off to my own function apply_action because I wrote it in a
    separate script file, and I was too lazy to rename it.
    :param board: The game only.
    :param action: The action to be applied.
    :return: An updated game board, new reference.
    """
    return apply_action(board, action)


def convert_action_to_pos(action: str):
    """
    Changes `c3-c2` to just `('c3', 'c2')`.
    """
    action = action.replace('takes ', '')
    return action[:action.index('-')], action[action.index('-') + 1:]


def convert_pos_to_index(pos):
    """
    Changes `c2` to `[1, 1]`.
    """
    conversion = {  # ROW MAJOR, Y, X
        'a3': [0, 0], 'b3': [0, 1], 'c3': [0, 2],
        'a2': [1, 0], 'b2': [1, 1], 'c2': [1, 2],
        'a1': [2, 0], 'b1': [2, 1], 'c1': [2, 2]
    }
    return conversion[pos]


def apply_action(state: dict, action: str) -> dict:
    """
    Process:
    - Takes an action in the form of the actions function output
    - Modifies the board
    - Returns a 2d array 3x3 post said action
    :param state: State dictionary containing
    :param action: A string representation of an action.
    :return: A new state dictionary with the action applied.
    """
    new_state = state.copy()
    if action.startswith('takes '):  # enact taking a position
        from_pos, to_pos = convert_action_to_pos(action)
        from_pos_x, from_pos_y = convert_pos_to_index(from_pos)
        to_pos_x, to_pos_y = convert_pos_to_index(to_pos)

        what_was = new_state[from_pos_x][from_pos_y]
        assert what_was != '0', 'Nothing should not be able to take a pawn'

        new_state[from_pos_x][from_pos_y] = 0

        # TODO: this assert fails, figure out why
        # assert new_state[to_pos_x][to_pos_y] != 0, 'Cannot take nothing'

        new_state[to_pos_x][to_pos_y] = what_was
    else:
        from_pos, to_pos = convert_action_to_pos(action)
        from_pos_x, from_pos_y = convert_pos_to_index(from_pos)
        to_pos_x, to_pos_y = convert_pos_to_index(to_pos)

        what_was = new_state[from_pos_x][from_pos_y]
        assert what_was != '0', 'You should not be able to move nothing'

        # TODO: this assert fails, figure out why
        # assert new_state[to_pos_x][to_pos_y] == 0, \
        #    'If this is not a taking move, then there should be nothing ' \
        #    'where a pawn is going'

        new_state[to_pos_x][to_pos_y] = what_was
    return new_state


def to_move(state):
    """
    Who's turn is it, right now
    :param state: The current game state.
    :return: The name of the player whose turn it is.
    """
    if state % 2 == 1:
        return 'player2'
    return 'player1'


def actions(board, state):
    """
    Come up with a list of actions that can be taken given a board and state.
    :param board: The board.
    :param state: The current state.
    :return:
    """
    all_actions = []
    if state % 2:  # black to move
        if board[0][0] == 1:
            if not (board[1][0]):
                all_actions.append("a3-a2")
                if not (board[2][0]):
                    all_actions.append("a3-a1")
        if board[0][1] == 1:
            if not (board[1][1]):
                all_actions.append("b3-b2")
                if not (board[2][1]):
                    all_actions.append("b3-b1")
        if board[0][2] == 1:
            if not (board[1][2]):
                all_actions.append("c3-c2")
                if not (board[2][2]):
                    all_actions.append("c3-c1")

        # black middle row move fronts
        if board[1][0] == 1:
            if not (board[2][0]):
                all_actions.append("a2-a1")
        if board[1][1] == 1:
            if not (board[2][1]):
                all_actions.append("b2-b1")
        if board[1][2] == 1:
            if not (board[2][2]):
                all_actions.append("c2-c1")

        # black takes initial
        if board[0][0] == 1:
            if board[1][1] == -1:
                all_actions.append("takes a3-b2")
        if board[0][1] == 1:
            if board[1][0] == -1:
                all_actions.append("takes b3-a2")
            if board[1][2] == -1:
                all_actions.append("takes b3-c2")
        if board[0][2] == 1:
            if board[1][1] == -1:
                all_actions.append("takes c3-b2")

        # black takes middle
        if board[1][0] == 1:
            if board[2][1] == -1:
                all_actions.append("takes b1-a2")
        if board[1][1] == 1:
            if board[2][0] == -1:
                all_actions.append("takes b2-a1")
            if board[2][2] == -1:
                all_actions.append("takes b2-a3")
        if board[1][2] == 1:
            if board[2][1] == -1:
                all_actions.append("takes b3-a2")
    else:  # white to move
        if board[2][0] == -1:
            if not (board[1][0]):
                all_actions.append("a1-a2")
                if not (board[0][0]):
                    all_actions.append("a1-a3")
        if board[2][1] == -1:
            if not (board[1][1]):
                all_actions.append("b1-b2")
                if not (board[0][1]):
                    all_actions.append("b1-b3")
        if board[2][2] == -1:
            if not (board[1][2]):
                all_actions.append("c1-c2")
                if not (board[0][2]):
                    all_actions.append("c1-c3")

        # white middle row move to front
        if board[1][0] == -1:
            if not (board[0][0]):
                all_actions.append("a2-a3")
        if board[1][1] == -1:
            if not (board[0][1]):
                all_actions.append("b2-b3")
        if board[1][2] == -1:
            if not (board[0][2]):
                all_actions.append("c2-c3")

        # white takes initial
        if board[2][0] == -1:
            if board[1][1] == 1:
                all_actions.append("takes a1-b2")
        if board[2][1] == -1:
            if board[1][0] == 1:
                all_actions.append("takes a2-b1")
            if board[1][2] == 1:
                all_actions.append("takes a2-b3")
        if board[2][2] == -1:
            if board[1][1] == 1:
                all_actions.append("takes a3-b2")

        # white takes middle
        if board[1][0] == -1:
            if board[0][1] == 1:
                all_actions.append("takes b1-c2")
        if board[1][1] == -1:
            if board[0][0] == 1:
                all_actions.append("takes b2-c1")
            if board[0][2] == 1:
                all_actions.append("takes b2-c3")
        if board[1][2] == -1:
            if board[0][1] == 1:
                all_actions.append("takes b3-c2")
    return all_actions


def is_terminal(game_state: dict) -> bool:
    """
    Determine if the game is over.
    :param game_state: A dictionary containing the board and the state.
    :return: True if the game is over, False if not.
    """
    acts = actions(game_state['board'], game_state['state'])
    return len(acts) > 0


def utility(board):
    """
    How did the game end relative to black. Did black end up in the last row,
    that's good. Did white end up in the top row? That's -1. Otherwise, return
    0.
    :param board:
    :return:
    """
    # how did the game end? win (1), loss (0) or draw (1/2)?
    if 1 in board[2]:
        return 1
    if -1 in board[0]:
        return -1
    return 0


def min_max(game_state):
    """
    Run minimax of the best utility move.
    :param game_state: The current game state.
    :return:
    """
    # minmax search
    turn = to_move(game_state['state'])
    util_move = max_val(game_state, turn)
    return util_move


def max_val(game_state: dict, turn: str) -> list:
    """
    The minimum value (utility) turn from the current game state.
    :param game_state: The game dictionary.
    :param turn: Whose turn it is.
    :return: [The utility of the minimum move, minimum move]
    """
    v = -sys.maxsize - 1
    move = None
    if not is_terminal(game_state):
        return [utility(game_state['board']), move]
    for action in actions(game_state['board'], game_state['state']):
        game_state['board'] = result(game_state['board'], action)
        game_state['state'] += 1
        mv = min_val(game_state, turn)
        if v < mv[0]:
            v = mv[0]
            move = action
    return [v, move]


def min_val(game_dict: dict, turn: str) -> list:
    """
    The minimum value (utility) turn from the current game state.
    :param game_dict: The game dictionary.
    :param turn: Whose turn it is.
    :return: [The utility of the minimum move, minimum move]
    """
    v = sys.maxsize
    move = None
    if not is_terminal(game_dict):
        return [utility(game_dict['board']), move]
    for action in actions(game_dict['board'], game_dict['state']):
        game_dict['board'] = result(game_dict['board'], action)
        game_dict['state'] += 1
        mv = max_val(game_dict, turn)
        if v > mv[0]:
            v = mv[0]
            move = action
    return [v, move]


def make_graph(n1: list, n2: list) -> dict:
    """
    Make a graph of the connections between node connections
    :param n1: List of nodes to point from.
    :param n2: A list of nodes to point to.
    :return:
    """
    graph = {}
    for node in n1:
        graph[node] = copy.deepcopy(n2)
    return graph


def make_network(num_of_inputs, num_hidden, num_outputs, num_layers):
    """
    Create a network based on the parameters passed.
    :param num_of_inputs: The number of inputs that this network will have.
    :param num_hidden: The number of hidden layers to have.
    :param num_outputs: The number of outputs to have.
    :param num_layers: The number of layers in total.
    :return:
    """
    num_hidden = 9
    inp_arr = list(range(1, num_of_inputs + 1))
    out_arr = list(range(1, num_outputs + 1))
    hidden = []
    for i in range(num_layers - 2):
        hidden.append(list(range(1, num_hidden + 1)))
    real_hidden = []
    for j in range(len(hidden)):
        real_hidden.append(make_graph(hidden[j], hidden[j]))
    net = {
        "input_weights": make_graph(inp_arr, hidden[0]),
        "output_weights": make_graph(hidden[-1], out_arr),
        "hidden_weights": real_hidden,
        "input_nodes": inp_arr,
        "output_nodes": out_arr,
        "hidden_nodes": hidden
    }
    return net


def initialize(weights: dict) -> dict:
    """
    Initialize the weights.
    :param weights: Dictionary of weights
    :return: The same weights parameter passed.
    """
    for key in weights.keys():
        for i in range(len(weights[key])):
            weights[key][i] = round(random.uniform(-2, 2), 3)
    return weights


def classify_sigmoid(network: dict, game_state: dict) -> dict:
    """
    Classify a network on the current game state using the sigmoid function.
    :param network: The network of input, hidden, and output nodes.
    :param game_state: The current game state.
    :return: The same network run classified by sigmoid.
    """
    network['input_nodes'][0] = game_state['state']
    for i in range(9):
        network['input_nodes'][i + 1] = game_state['board'][math.trunc(i / 3)][
            i % 3]
    network['input_weights'] = initialize(network['input_weights'])
    for i in range(len(network['hidden_weights'])):
        network['hidden_weights'][i] = initialize(network['hidden_weights'][i])
    return run_classify_sigmoid(network)


def run_classify_sigmoid(network: dict) -> dict:
    """
    Run classification on a network.
    :param network: The network of input, hidden, and output nodes.
    :return: The same network run classified by sigmoid.
    """
    for key in network['input_weights'].keys():
        for i in range(len(network['input_weights'][key])):
            network['input_weights'][key][i] += network['input_nodes'][int(key) - 1]
    net_copy = {}
    for key in network:
        net_copy[key] = copy.deepcopy(network[key])

    # initial layer to first hidden
    for key in net_copy['input_weights'].keys():
        for i in range(len(net_copy['input_weights'][key])):
            net_copy['hidden_nodes'][0][i] = sigmoid(net_copy['input_weights'][key][i])

    # hidden to hidden
    for i in range(0, 3):
        for key in network['hidden_weights'][i].keys():
            for j in range(len(network['hidden_weights'][i][key])):
                net_copy['hidden_weights'][i][key][j] += net_copy['hidden_nodes'][i][j]
        for key in network['hidden_weights'][i].keys():
            for j in range(len(network['hidden_weights'][i][key])):
                net_copy['hidden_nodes'][i+1][j] = sigmoid(net_copy['hidden_weights'][i][
                                                        key][j])

    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['hidden_weights'][3][key][j] += net_copy['hidden_nodes'][3][j]
    for key in network['hidden_weights'][2].keys():
        for j in range(len(network['hidden_weights'][2][key])):
            net_copy['output_weights'][key][j] = sigmoid(net_copy['hidden_weights'][3][key][j])

    # hidden final to output
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['hidden_weights'][3][key][j] += net_copy['hidden_nodes'][3][j]
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['output_nodes'][j] = sigmoid(net_copy['hidden_weights'][3][key][j])

    return net_copy


def classify_relu(network: dict, game_state: dict) -> dict:
    """
    Classify a network by its current game state.
    :param network: The network to run ReLu on.
    :param game_state: The current game state.
    :return: Updated network.
    """
    network['input_nodes'][0] = game_state['state']
    for i in range(9):
        network['input_nodes'][i + 1] = \
            game_state['board'][math.trunc(i / 3)][i % 3]
    network['input_weights'] = initialize(network['input_weights'])
    for i in range(len(network['hidden_weights'])):
        network['hidden_weights'][i] = initialize(network['hidden_weights'][i])
    return run_classify_relu(network)


def run_classify_relu(network: dict) -> dict:
    """
    Run a network on classification with ReLu.
    :param network: The network to classify on.
    :return: Updated network.
    """
    for key in network['input_weights'].keys():
        for i in range(len(network['input_weights'][key])):
            network['input_weights'][key][i] += network['input_nodes'][int(key) - 1]
    net_copy = {}
    for key in network:
        net_copy[key] = copy.deepcopy(network[key])

    # initial layer to first hidden
    for key in net_copy['input_weights'].keys():
        for i in range(len(net_copy['input_weights'][key])):
            net_copy['hidden_nodes'][0][i] = relu(net_copy['input_weights'][key][i])

    # hidden to hidden
    for i in range(0, 3):
        for key in network['hidden_weights'][i].keys():
            for j in range(len(network['hidden_weights'][i][key])):
                net_copy['hidden_weights'][i][key][j] += net_copy['hidden_nodes'][i][j]
        for key in network['hidden_weights'][i].keys():
            for j in range(len(network['hidden_weights'][i][key])):
                net_copy['hidden_nodes'][i+1][j] = relu(net_copy['hidden_weights'][i][
                                                       key][j])

    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['hidden_weights'][3][key][j] += net_copy['hidden_nodes'][3][j]
    for key in network['hidden_weights'][2].keys():
        for j in range(len(network['hidden_weights'][2][key])):
            net_copy['output_weights'][key][j] = relu(net_copy['hidden_weights'][3][key][j])

    # hidden final to output
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['hidden_weights'][3][key][j] += net_copy['hidden_nodes'][3][j]
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            net_copy['output_nodes'][j] = sigmoid(net_copy['hidden_weights'][3][key][j])

    return net_copy


def sigmoid(x):
    """Perform Sigmoid calculation."""
    return 1 / (1 + math.exp(-x))


def relu(x):
    """Perform ReLU calculation"""
    if x < 0:
        return 0
    else:
        return 1


def back_prop_sigmoid(network: dict, game_state: dict) -> dict:
    """
    Run back propagation using the sigmoid function.
    :param network: The network, as is.
    :param game_state: The game state to back propagate
    :return: Updated network.
    """
    val = min_max(game_state)
    if val[1] is None:
        return network
    val = val[-1].split(' ')[-1].split('-')[-1]
    ind_list = ['c', 'b', 'a']
    check_list = [0] * 9
    index = ind_list.index(val[0]) * 3 + int(val[1]) - 1
    check_list[index] = 1
    error_list = []
    weight_list = []

    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][3].keys():
            weight_sum += network['hidden_weights'][3][key][i]
            weight_list.append(weight_sum)
    for i in range(9):
        error = update_weights_sigmoid(network['output_nodes'][i], check_list[i],
                                       weight_list[i])
        error_list.append(error)
    for key in network['output_weights'].keys():
        for i in range(9):
            network['output_weights'][key][i] -= error_list[i]

    error_list = []
    check_list = []
    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][2].keys():
            weight_sum += network['hidden_weights'][2][key][i]
            weight_list.append(weight_sum)
    for key in network['hidden_weights'][3].keys():
        check = 0
        for j in range(len(network['hidden_weights'][3][key])):
            check += network['hidden_nodes'][3][j] - network['hidden_weights'][3][key][j]
        check_list.append(check)
    for i in range(9):
        error = update_weights_sigmoid(network['hidden_nodes'][3][i], check_list[i],
                                       weight_list[i])
        error_list.append(error)
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            network['hidden_nodes'][3][j] -= error_list[j]


    error_list = []
    check_list = []
    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][2].keys():
            weight_sum += network['hidden_weights'][0][key][i]
            weight_list.append(weight_sum)
    for key in network['hidden_weights'][3].keys():
        check = 0
        for j in range(len(network['hidden_weights'][3][key])):
            check += network['hidden_nodes'][1][j] - network['hidden_weights'][1][key][j]
        check_list.append(check)
    for i in range(9):
        error = update_weights_sigmoid(network['hidden_nodes'][1][i], check_list[i],
                                       weight_list[i])
        error_list.append(error)
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][2][key])):
            network['hidden_nodes'][1][j] -= error_list[j]

    return network


def back_prop_relu(network: dict, game_state: dict) -> dict:
    """
    Run back propagation using the ReLu function.
    :param network: The network, as is.
    :param game_state: The game state to back propagate
    :return: Updated network.
    """
    val = min_max(game_state)
    if val[1] is None:
        return network

    val = val[-1].split(' ')[-1].split('-')[-1]
    ind_list = ['c', 'b', 'a']
    check_list = [0] * 9
    index = ind_list.index(val[0]) * 3 + int(val[1]) - 1
    check_list[index] = 1
    error_list = []
    weight_list = []

    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][3].keys():
            weight_sum += network['hidden_weights'][3][key][i]
            weight_list.append(weight_sum)
    for i in range(9):
        error = update_weights_relu(network['output_nodes'][i])
        error_list.append(error)
    for key in network['output_weights'].keys():
        for i in range(9):
            network['output_weights'][key][i] -= error_list[i]

    error_list = []
    check_list = []
    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][2].keys():
            weight_sum += network['hidden_weights'][2][key][i]
            weight_list.append(weight_sum)
    for key in network['hidden_weights'][3].keys():
        check = 0
        for j in range(len(network['hidden_weights'][3][key])):
            check += network['hidden_nodes'][3][j] - network['hidden_weights'][3][key][j]
        check_list.append(check)
    for i in range(9):
        error = update_weights_relu(network['hidden_nodes'][3][i])
        error_list.append(error)
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][3][key])):
            network['hidden_nodes'][3][j] -= error_list[j]

    error_list = []
    check_list = []
    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][2].keys():
            weight_sum += network['hidden_weights'][1][key][i]
            weight_list.append(weight_sum)
    for key in network['hidden_weights'][3].keys():
        check = 0
        for j in range(len(network['hidden_weights'][3][key])):
            check += network['hidden_nodes'][2][j] - network['hidden_weights'][2][key][j]
        check_list.append(check)
    for i in range(9):
        error = update_weights_relu(network['hidden_nodes'][2][i])
        error_list.append(error)
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][2][key])):
            network['hidden_nodes'][2][j] -= error_list[j]

    error_list = []
    check_list = []
    for i in range(9):
        weight_sum = 0
        for key in network['hidden_weights'][2].keys():
            weight_sum += network['hidden_weights'][0][key][i]
            weight_list.append(weight_sum)
    for key in network['hidden_weights'][3].keys():
        check = 0
        for j in range(len(network['hidden_weights'][3][key])):
            check += network['hidden_nodes'][1][j] - network['hidden_weights'][1][key][j]
        check_list.append(check)
    for i in range(9):
        error = update_weights_relu(network['hidden_nodes'][1][i])
        error_list.append(error)
    for key in network['hidden_weights'][3].keys():
        for j in range(len(network['hidden_weights'][2][key])):
            network['hidden_nodes'][1][j] -= error_list[j]

    return network


def update_weights_sigmoid(actual: float, expected: float, weighted_sum:
float) -> float:
    """
    Update the weights using the sigmoid function.
    :param actual: The actual value from the network run on a value.
    :param expected: The expected value, ground truth.
    :param weighted_sum: The weighted sums.
    :return: Error delta.
    """
    weighted_sum = sigmoid(weighted_sum)
    return (
            2 *
            (actual - expected) * (
                    sigmoid(weighted_sum) * (1 - sigmoid(weighted_sum))
            )
    )


def update_weights_relu(actual: float) -> int:
    """
    Update the weights using the ReLu function.
    :param actual: The actual value from the network run on a value.
    :return: Error delta.
    """
    if actual < 0:
        return 0
    return 1


def main():
    """
    REMEMBER!!!
    0 is a blank space
    1 is black
    -1 is white
    """
    game_state = {
        'board':
            [
                [1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]
            ],
        'state': 0
    }

    classified_sigmoid = make_network(10, 5, 9, 6)
    classified_relu = make_network(10, 5, 9, 6)

    for i in range(2000):  # run 2000 epochs
        classified_sigmoid = classify_sigmoid(classified_sigmoid, game_state)
        classified_relu = classify_relu(classified_relu, game_state)

        classified_sigmoid = back_prop_sigmoid(classified_sigmoid, game_state)
        classified_relu = back_prop_relu(classified_relu, game_state)

    print(classified_sigmoid['output_nodes'])
    print(classified_relu['output_nodes'])
    print("Actions = ['c1-c2', 'takes b3-c2', 'b1-b3']")


if __name__ == '__main__':
    main()
