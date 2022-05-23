testDict = {
    'board': [
        [1, 1, 1], # -1 = white
        [0, -1, -1], # 1 = black
        [-1, 0, 0]
    ],
    'state': 0
}


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
        if board[1][0] == - 1:
            if not (board[0][0]):
                legals.append("a2-a3")
        if board[1][1] == -1:
            if not (board[0][1]):
                legals.append("b2-a3")
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


def convert_action_to_pos(action: str):
    """
    Changes 'c3-c2' to just ('c3', 'c2')
    """
    action = action.replace('takes ', '')
    return action[:action.index('-')], action[action.index('-') + 1:]


def convert_pos_to_index(pos):
    """
    Changes 'c2' to [1, 1]
    """
    conversion = { # ROW MAJOR, Y, X
        'a3': [0, 0], 'b3': [0, 1], 'c3': [0, 2],
        'a2': [1, 0], 'b2': [1, 1], 'c2': [1, 2],
        'a1': [2, 0], 'b1': [2, 1], 'c1': [2, 2]
    }
    return conversion[pos]


def apply_action(state, action: str):
    """
    - Takes an action in the form of the actions function output
    - Modifies the board
    - Returns a 2d array 3x3 post said action
    :param state: State dictionary containing
    :param action:
    :return:
    """
    new_state = state.copy()
    if action.startswith('takes '):  # enact taking a position
        from_pos, to_pos = convert_action_to_pos(action)
        from_pos_x, from_pos_y = convert_pos_to_index(from_pos)
        to_pos_x, to_pos_y = convert_pos_to_index(to_pos)

        what_was = new_state['board'][from_pos_x][from_pos_y]
        assert what_was != '0', 'Nothing should not be able to take a pawn'

        new_state['board'][from_pos_x][from_pos_y] = 0

        assert new_state['board'][to_pos_x][to_pos_y] != 0, \
            'Cannot take nothing'

        new_state['board'][to_pos_x][to_pos_y] = what_was
    else:
        from_pos, to_pos = convert_action_to_pos(action)
        from_pos_x, from_pos_y = convert_pos_to_index(from_pos)
        to_pos_x, to_pos_y = convert_pos_to_index(to_pos)

        what_was = new_state['board'][from_pos_x][from_pos_y]
        assert what_was != '0', 'You should not be able to move nothing'

        assert new_state['board'][to_pos_x][to_pos_y] == 0, \
            'If this is not a taking move, then there should be nothing ' \
            'where a pawn is going'

        new_state['board'][to_pos_x][to_pos_y] = what_was
    new_state['state'] += 1
    return new_state


def apply_action_2(board, action):
    action = action.split(' ')[-1].split('-')
    rows = ['a', 'b', 'c']
    cols = ['3', '2', '1']
    print(action)
    curRow = int(rows.index(action[0][0]))
    curCol = int(cols.index(action[0][1]))
    modRow = int(rows.index(action[1][0]))
    modCol = int(cols.index(action[1][1]))
    inVal = board[curCol][curRow]
    board[curCol][curRow] = 0
    board[modCol][modRow] = inVal
    return board


def pretty_print_2d(arr) -> None:
    """Print a 2D array with tabs in between."""
    print('\n'.join(['\t\t'.join([str(cell) for cell in row]) for row in arr]))


if __name__ == '__main__':
    pretty_print_2d(testDict['board'])
    updated_board = apply_action(testDict, 'takes b2-c3')
    pretty_print_2d(updated_board['board'])
