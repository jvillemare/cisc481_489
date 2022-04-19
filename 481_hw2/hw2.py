"""
hw2.py
James Villemarette
Due April 19, 2022
CISC 489 - 010

Layout:
 - imports
 - initialize puzzles
 - template setup
 - program writeup functions
 - routes
 - main
"""

# imports
from flask import Flask, send_file
from sudoku_constraints import constraints_9x9
from four_constraints import four_constraints
from four_domains import four_domains
from typing import List, Dict
from string import Template
import os
import math

# initialize puzzles
puzzle_1 = [
    [7, 0, 0, 4, 0, 0, 0, 8, 6],
    [0, 5, 1, 0, 8, 0, 4, 0, 0],
    [0, 4, 0, 3, 0, 7, 0, 9, 0],
    [3, 0, 9, 0, 0, 6, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 4, 9, 0, 0, 7, 0, 8],
    [0, 8, 0, 1, 0, 2, 0, 6, 0],
    [0, 0, 6, 0, 5, 0, 9, 1, 0],
    [2, 1, 0, 0, 0, 3, 0, 0, 5]
]

puzzle_2 = [
    [1, 0, 0, 2, 0, 3, 8, 0, 0],
    [0, 8, 2, 0, 6, 0, 1, 0, 0],
    [7, 0, 0, 0, 0, 1, 6, 4, 0],
    [3, 0, 0, 0, 9, 5, 0, 2, 0],
    [0, 7, 0, 0, 0, 0, 0, 1, 0],
    [0, 9, 0, 3, 1, 0, 0, 0, 6],
    [0, 5, 3, 6, 0, 0, 0, 0, 1],
    [0, 0, 7, 0, 2, 0, 3, 9, 0],
    [0, 0, 4, 1, 0, 9, 0, 0, 5]
]

puzzle_3 = [
    [1, 0, 0, 8, 4, 0, 0, 5, 0],
    [5, 0, 0, 9, 0, 0, 8, 0, 3],
    [7, 0, 0, 0, 6, 0, 1, 0, 0],
    [0, 1, 0, 5, 0, 2, 0, 3, 0],
    [0, 7, 5, 0, 0, 0, 2, 6, 0],
    [0, 3, 0, 6, 0, 9, 0, 4, 0],
    [0, 0, 7, 0, 5, 0, 0, 0, 6],
    [4, 0, 1, 0, 0, 6, 0, 0, 7],
    [0, 6, 0, 0, 9, 4, 0, 0, 2]
]

puzzle_4 = [
    [0, 0, 0, 0, 9, 0, 0, 7, 5],
    [0, 0, 1, 2, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 1, 8, 0],
    [3, 0, 0, 6, 0, 0, 9, 0, 0],
    [1, 0, 0, 0, 5, 0, 0, 0, 4],
    [0, 0, 6, 0, 0, 2, 0, 0, 3],
    [0, 3, 2, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 6, 5, 0, 0],
    [7, 9, 0, 0, 1, 0, 0, 0, 0]
]

puzzle_5 = [
    [0, 0, 0, 0, 0, 6, 0, 8, 0],
    [3, 0, 0, 0, 0, 2, 7, 0, 0],
    [7, 0, 5, 1, 0, 0, 6, 0, 0],
    [0, 0, 9, 4, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 9, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 8, 3, 0, 0],
    [0, 0, 4, 0, 0, 7, 8, 0, 5],
    [0, 0, 2, 8, 0, 0, 0, 0, 6],
    [0, 5, 0, 9, 0, 0, 0, 0, 0]
]

builtin_puzzles = [
    puzzle_1,
    puzzle_2,
    puzzle_3,
    puzzle_4,
    puzzle_5
]


def validate_board(board: List[List[int]], length: int) -> bool:
    """
    Check if a board is valid.
    :param board: The board itself.
    :param length: The length of a Sudoku board, as to represent
        (length x length).
    :return: True if it is a (length x length; 9x9; etc.), False if not.
    """
    # check y
    if len(board) != length:
        return False
    # check x
    for i in range(length):
        if len(board[i]) != length:
            return False
    return True


for b in builtin_puzzles:
    assert validate_board(b, 9), "One of these boards is not a true 9x9"


# template setup
class SpecialTemplate(Template):
    """
    Update regular `string.Template` to use a new delimiter, `~`, and make the
    `idpattern` more narrow (`a-z`, `_`, and `0-9`).
    """
    delimiter = '~'
    idpattern = r'[a-z][_a-z0-9]*'


def load_template(filepath: str) -> SpecialTemplate:
    """
    Read a file into a `string.Template` Python built-in class.

    Args:
     - filepath: The path to a template

    Returns: A loaded template.
    """
    t = 0
    assert os.path.isfile(filepath), 'Provided template is not a valid filepath'
    with open(filepath, 'r', encoding='utf-8') as f:
        t = SpecialTemplate('\n'.join(f.readlines()))
    return t


preview_base_temp = load_template('static/preview_base.html')


# program writeup functions
class CSP:
    """
    Holds the cool stuff for the Constraint Satisfaction Problem (CSP)
    """

    domains: Dict[str, List[int]]
    constraints: Dict[set, List[int]]
    variables: List[int]
    neighbors: Dict[str, List[str]]

    def __init__(self, domains, constraints, size: int):
        """
        Constructor.
        :param domains:
        :param constraints:
        :param size: The size of a board, (size x size), like 9 for 9x9, or
            4 for 4x4.
        """
        self.domains = domains
        self.constraints = constraints
        self.size = size
        # variables are literally just all the cells in the puzzle
        self.variables = []
        for row in range(size + 1):
            for col in range(size + 1):
                self.variables.append('C' + str(row) + str(col))
        self.neighbors = {}
        for row in range(size + 1):
            for col in range(size + 1):
                self.neighbors['C' + str(row) + str(col)] = None

    def get_neighbors_column(self, x):
        col = []
        for c in range(1, self.size + 1):
            col.append('C' + str(c) + str(x))
        return col

    def get_neighbors_row(self, y):
        row = []
        for r in range(1, self.size + 1):
            row.append('C' + str(y) + str(r))
        return row

    def get_neighbors_square(self, cell_id):
        """
        Get the neighbors in a square for a cell. Yes this is super hacky. Yes,
        I do not care.
        :param cell_id:
        :return:
        """
        if self.size == 4:
            top_left = [
                'C11', 'C12',
                'C21', 'C22'
            ]
            top_right = [
                'C13', 'C14',
                'C23', 'C24'
            ]
            bottom_left = [
                'C31', 'C32',
                'C41', 'C42'
            ]
            bottom_right = [
                'C33', 'C34',
                'C43', 'C44'
            ]
            resolution = {
                'C11': top_left,
                'C12': top_left,
                'C21': top_left,
                'C22': top_left,
                'C13': top_right,
                'C14': top_right,
                'C23': top_right,
                'C24': top_right,
                'C31': bottom_left,
                'C32': bottom_left,
                'C41': bottom_left,
                'C42': bottom_left,
                'C33': bottom_right,
                'C34': bottom_right,
                'C43': bottom_right,
                'C44': bottom_right
            }
            return resolution[cell_id]
        else:
            raise 'Unsupported Sudoku board size, only 4x4, 9x9, sorry not sorry'

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""
        count = 0
        for var2 in self.neighbors.get(var):
            val2 = None
            if assignment.__contains__(var2):
                val2 = assignment[var2]
            if val2 is not None and self.constraints(var, val, var2, val2) is False:
                count += 1
        return count

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))


def revise(csp, Xi, Xj) -> bool:
    """
    ...?
    :param csp:
    :param Xi:
    :param Xj:
    :return:
    """
    revised = False
    for x in csp.domains[Xi]:
        if all(not x != y for y in csp.domains[Xj]):
            del csp.domains[Xi][csp.domains[Xi].index(x)]
            revised = True
    return revised


def ac3(csp) -> bool:
    """
    Takes as input a CSP and modifies it such that any incon- sistent values
    across all domains are removed. The function returns a boolean indicating
    whether or not all variables have at least on value left in their domains.
    :param csp:
    :return:
    """
    queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
    #csp.support_pruning()
    while queue:
        (Xi, Xj) = queue.pop()
        if revise(csp, Xi, Xj):
            if not csp.domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xi:
                    queue.append((Xk, Xi))
    return True


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.domains[var])
    else:
        count = 0
        for val in csp.domains[var]:
            # TODO: nconflicts?
            if csp.nconflicts(var, val, assignment) == 0:
                count += 1
        return count


def minimum_remaining_values(csp, assignment):
    """
    Takes a CSP and a set of variable assignments as input, and returns the
    variable with the fewest values in its domain among the unassigned variables
    in the CSP.
    :param csp:
    :param assignment:
    :return:
    """
    vars_to_check = []
    size = []
    for v in csp.variables:
        if v not in assignment.keys():
            vars_to_check.append(v)
            size.append(num_legal_values(csp, v, assignment))
    return vars_to_check[size.index(min(size))]


def backtracking_search(csp):
    """
    Takes a CSP and finds a valid assignment for all the variables in the CSP,
    if one exists. It should leverage your AC-3 implementation to maintain arc
    consistency.1 When Choosing a variable to assign, it should use your minimum
    remaining values heuristic implementation. Along with the solution to the
    CSP, your search should return the order in which it assigned the variables,
    and the domains of the remaining unassigned variables after each assignment.
    :param csp:
    :return:
    """
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    else:
                        csp.n_bt += 1
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result2 = backtrack({})
    assert result2 is None or csp.goal_test(result2)
    return result2


# routes
app = Flask('cisc489_sudoku', static_url_path="/static/",
            static_folder="static")


def convert_board_to_html(board: List[List[int]]) -> str:
    """
    Convert a Sudoku board to HTML.
    :param board: Board as a list of list of ints.
    :return: HTML of the board
    """
    html = '<table>\n'
    thick_black_lines = math.floor(math.sqrt(len(board)))
    for cg in range(thick_black_lines):
        html += '\t<colgroup>' + ('<col>' * thick_black_lines) + '\n'
    tb_counter = 0
    for row in range(len(board)):
        if tb_counter == 0:
            html += '\t<tbody>'
        html += '\t\t<tr> '
        for horizontal in range(len(board)):
            if board[row][horizontal] == 0:
                html += '<td> </td> '
            else:
                html += '<td>' + str(board[row][horizontal]) + '</td> '
        html += '</tr>'
        tb_counter += 1
        if tb_counter == thick_black_lines:
            html += '\n\t</tbody>'
            tb_counter = 0
    return html


def convert_string_to_board(board: str) -> List[List[int]]:
    """
    Convert a string to a board, where the string looks like
    ```
    12,
    34
    ```
    or
    ```
    12,
    3x
    ```
    with no new lines. And the `x` represents an empty space.
    :param board: The board as a string, n x n size.
    :return: The board as a list of lists.
    """
    v_board = []
    rows = board.split(',')
    for r in rows:
        v_row = []
        for char in r:
            v_row.append(char)
        v_board.append(v_row)
    return v_board


@app.route('/')
def index_page():
    """
    Serves the home page.
    """
    return send_file('static/index.html')


@app.route('/input/')
def input_page():
    """
    Serves the input page, which lets users manually type in a Sudoku puzzle.
    """
    return send_file('static/input.html')


@app.route('/builtin/')
def index_builtin():
    """
    Shows all the built-in puzzles.
    """
    gen_puzzle_list = '<ul>\n'
    for i in range(len(builtin_puzzles)):
        gen_puzzle_list += '\t<li>puzzle_' + str(i+1) + '</li>\n'
        gen_puzzle_list += '\t<ul>\n'
        gen_puzzle_list += f'\t\t<li><a href="/builtin/display/{i}">Preview puzzle</a></li>\n'
        gen_puzzle_list += f'\t\t<li><a href="/builtin/solve/{i}">Solve puzzle</a></li>\n'
        gen_puzzle_list += '\t</ul>'
    gen_puzzle_list += '</ul>\n'
    return gen_puzzle_list


@app.route('/builtin/display/<int:puzzle_num>')
def display_builtin_puzzle(puzzle_num: int):
    """
    Just displays a built-in puzzle, does not solve it. Used for debugging.
    :param puzzle_num: The number of the puzzle
    :return:
    """
    return preview_base_temp.substitute(
        preview = convert_board_to_html(
            builtin_puzzles[puzzle_num]
        )
    )


@app.route('/builtin/solve/<int:puzzle_num>')
def solve_builtin_puzzle(puzzle_num: int):
    """
    Solves a built-in puzzle.
    :param puzzle_num:
    :return:
    """
    return ""


@app.route('/display/<board>')
def display_provided_puzzle(board: str):
    """
    Display a puzzle that was provided by the user.
    :param board: The board as a string, "123456789,123456789,..."
    :return:
    """
    return convert_board_to_html(
        convert_string_to_board(board)
    )


@app.route('/solve/<board>')
def solve_provided_puzzle(board: str):
    """
    Solve a puzzle that was passed in by the URL.
    :param board:
    :return:
    """
    return ""


# main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5003", debug=True)
