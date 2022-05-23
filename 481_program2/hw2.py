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
puzzle_1 = [  # provided by professor in program2-writeup.pdf
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

puzzle_2 = [  # provided
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

puzzle_3 = [  # provided
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

puzzle_4 = [  # provided
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

puzzle_5 = [  # provided
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

puzzle_6 = [  # simple 4x4
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
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


def convert_str_to_pos(s: str) -> List[int]:
    """Convert a string representation of a cell to a list of int coordinates"""
    s = s.replace('C', '')
    return [int(s[0]) - 1, int(s[1]) - 1]


def convert_pos_to_str(pos: List[int]) -> str:
    """Convert a list of int coordinates of a cell to a string representation"""
    pos_x = pos[0] + 1
    pos_y = pos[1] + 1
    return 'C' + str(pos_x) + str(pos_y)


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

    variables: List[str]
    domains: Dict[str, List[int]]
    constraints: Dict[set, List[int]]
    variables: List[int]
    neighbors: Dict[str, List[str]]

    def __init__(self, domains, neighbors, constraints, size: int):
        """
        Constructor.
        :param domains:
        :param constraints:
        :param size: The size of a board, (size x size), like 9 for 9x9, or
            4 for 4x4.
        """
        self.variables = domains.keys()
        self.domains = domains
        # TODO: maybe deepcopy()?
        self.original_domains = domains.copy()
        self.constraints = constraints
        self.size = size
        self.assign_counter = 0
        # variables are literally just all the cells in the puzzle
        """
        self.variables = []
        for row in range(size + 1):
            for col in range(size + 1):
                self.variables.append('C' + str(row) + str(col))
        """
        self.neighbors = neighbors

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.domains is None:
            self.domains = {v: list(self.original_domains[v]) for v in
                                 self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.domains[var] if a != value]
        self.domains[var] = [value]
        return removals

    def nconflicts(self, var, val, assignment):
        """
        Return the number of conflicts var=val has with other variables.
        """
        count = 0
        for var2 in self.neighbors.get(var):
            val2 = None
            if assignment.__contains__(var2):
                val2 = assignment[var2]
            if val2 is not None and val != val2 is False:
                count += 1
        return count

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.assign_counter += 1

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.domains or self.original_domains)[var]

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.domains[B].append(b)

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def goal_test(self, state):
        """
        The goal is to assign all variables, with all constraints satisfied.
        """
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


class SudokuCSP(CSP):

    def __init__(self, board):

        self.domains = {}
        self.neighbors = {}
        self.size = len(board)
        # our variables will be named as "CELL NUMBER"
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                self.neighbors.update({'C' + str(x) + str(y): {}})
        for x in range(self.size):
            for y in range(self.size):
                var = 'C' + str(x) + str(y)
                self.add_neighbor(var, self.get_neighbors_row(y) |
                                  self.get_neighbors_column(x) |
                                  self.get_neighbors_square(x, y))
                # if the board has a value in cell[i][j] the domain of this
                # variable will be that number
                if board[x][y] != 0:
                    self.domains.update({var: [board[x][y]]})
                else:
                    self.domains.update({var: list(range(1, self.size + 1))})

        c = None
        if self.size == 9:
            c = constraints_9x9
        elif self.size == 4:
            c = four_constraints
        else:
            raise 'Nonoonononooo not supported, never will be supported'
        CSP.__init__(self,
                     self.domains,
                     self.neighbors,
                     c,
                     self.size)

    def get_neighbors_column(self, x):
        col = set()
        x += 1
        for c in range(1, self.size + 1):
            col.add('C' + str(c) + str(x))
        return col

    def get_neighbors_row(self, y):
        row = set()
        y += 1
        for r in range(1, self.size + 1):
            row.add('C' + str(y) + str(r))
        return row

    def get_neighbors_square(self, cell_x, cell_y):
        """
        Get the neighbors in a square for a cell. Yes this is super hacky. Yes,
        I do not care.
        :param cell_id:
        :return:
        """
        cell_id = convert_pos_to_str([cell_x, cell_y])
        #print('get_neighbors_square', cell_x, cell_y)
        #print(cell_id)
        if self.size == 4:  # 4x4 neighbors
            top_left = set([
                'C11', 'C12',
                'C21', 'C22'
            ])
            top_right = set([
                'C13', 'C14',
                'C23', 'C24'
            ])
            bottom_left = set([
                'C31', 'C32',
                'C41', 'C42'
            ])
            bottom_right = set([
                'C33', 'C34',
                'C43', 'C44'
            ])
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
        elif self.size == 9:  # 9x9 neighbors
            top_left = set([
                'C11', 'C12', 'C13',
                'C21', 'C22', 'C23',
                'C31', 'C32', 'C33'
            ])
            top_middle = set([
                'C14', 'C15', 'C16',
                'C24', 'C25', 'C26',
                'C34', 'C35', 'C36'
            ])
            top_right = set([
                'C17', 'C18', 'C19',
                'C27', 'C28', 'C29',
                'C37', 'C38', 'C39'
            ])
            middle_left = set([
                'C41', 'C42', 'C43',
                'C51', 'C52', 'C53',
                'C61', 'C62', 'C63'
            ])
            middle_middle = set([
                'C44', 'C45', 'C46',
                'C54', 'C55', 'C56',
                'C64', 'C65', 'C66'
            ])
            middle_right = set([
                'C47', 'C48', 'C49',
                'C57', 'C58', 'C59',
                'C67', 'C68', 'C69'
            ])
            bottom_left = set([
                'C71', 'C72', 'C73',
                'C81', 'C82', 'C83',
                'C91', 'C92', 'C93'
            ])
            bottom_middle = set([
                'C74', 'C75', 'C76',
                'C84', 'C85', 'C86',
                'C94', 'C95', 'C96'
            ])
            bottom_right = set([
                'C77', 'C78', 'C79',
                'C87', 'C88', 'C89',
                'C97', 'C98', 'C99'
            ])
            resolution = {
                # top_left
                'C11': top_left, 'C12': top_left, 'C13': top_left,
                'C21': top_left, 'C22': top_left, 'C23': top_left,
                'C31': top_left, 'C32': top_left, 'C33': top_left,
                # top_middle
                'C14': top_middle, 'C15': top_middle, 'C16': top_middle,
                'C24': top_middle, 'C25': top_middle, 'C26': top_middle,
                'C34': top_middle, 'C35': top_middle, 'C36': top_middle,
                # top_right
                'C17': top_right, 'C18': top_right, 'C19': top_right,
                'C27': top_right, 'C28': top_right, 'C29': top_right,
                'C37': top_right, 'C38': top_right, 'C39': top_right,
                # middle_left
                'C41': middle_left, 'C42': middle_left, 'C43': middle_left,
                'C51': middle_left, 'C52': middle_left, 'C53': middle_left,
                'C61': middle_left, 'C62': middle_left, 'C63': middle_left,
                # middle_middle
                'C44': middle_middle, 'C45': middle_middle,
                'C46': middle_middle,
                'C54': middle_middle, 'C55': middle_middle,
                'C56': middle_middle,
                'C64': middle_middle, 'C65': middle_middle,
                'C66': middle_middle,
                # middle_right
                'C47': middle_right, 'C48': middle_right, 'C49': middle_right,
                'C57': middle_right, 'C58': middle_right, 'C59': middle_right,
                'C67': middle_right, 'C68': middle_right, 'C69': middle_right,
                # bottom_left
                'C71': bottom_left, 'C72': bottom_left, 'C73': bottom_left,
                'C81': bottom_left, 'C82': bottom_left, 'C83': bottom_left,
                'C91': bottom_left, 'C92': bottom_left, 'C93': bottom_left,
                # bottom_middle
                'C74': bottom_middle, 'C75': bottom_middle,
                'C76': bottom_middle,
                'C84': bottom_middle, 'C85': bottom_middle,
                'C86': bottom_middle,
                'C94': bottom_middle, 'C95': bottom_middle,
                'C96': bottom_middle,
                # bottom_right
                'C77': bottom_right, 'C78': bottom_right, 'C79': bottom_right,
                'C87': bottom_right, 'C88': bottom_right, 'C89': bottom_right,
                'C97': bottom_right, 'C98': bottom_right, 'C99': bottom_right
            }
            return resolution[cell_id]
        else:
            raise 'Unsupported Sudoku board size, only 4x4, 9x9, sorry not sorry'

    def add_neighbor(self, var, elements):
        # we dont want to add variable as its self neighbor
        self.neighbors.update({var: {x for x in elements if x != var}})


def revise_check(i, c1, c2, domain):
    for j in domain:
        if c1 is not None and [i, j] in c1 or c2 is not None and [i, j] in c2:
            return True
    return False


def revise_old(csp: CSP, x_i, x_j) -> bool:
    initial = len(csp.domains[x_i])

    c1 = csp.constraints.get((x_i, x_j))
    c2 = csp.constraints.get((x_j, x_i))
    csp.domains[x_i][:] = [
        i for i in csp.domains[x_i] if revise_check(i, c1, c2, csp.domains[x_j])
    ]
    return initial != len(csp.domains[x_i])


def revise(csp: CSP, x_i, x_j) -> bool:
    """
    ...?
    :param csp: The CSP problem to search the domains in.
    :param x_i:
    :param x_j:
    :return:
    """
    revised = False
    if x_i not in csp.domains:
        return revised
    for x in csp.domains[x_i]:
        # TODO: logic may not work here
        # TODO: make sure not the case that all of them are equal to x
        # TODO: if i can't find a pair, then "i'm not good"
        # Prof. Keffer: looks good! maybe, double check
        if x_j not in csp.domains:
            continue
        if all(not x != y for y in csp.domains[x_j]):
            del csp.domains[x_i][csp.domains[x_i].index(x)]
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
    # or?
    # queue = [(X, var) for X in csp.neighbors[var]]
    queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
    #csp.support_pruning()
    while queue:
        (Xi, Xj) = queue.pop()
        # Don't forget that if revise is true, we know Xi was revised
        # Prof. Keffer: "you're right!"
        if revise(csp, Xi, Xj):
            if not csp.domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                # TODO: maybe this Xi should be Xj
                if Xk != Xi:
                    queue.append((Xk, Xi))
    return True


def num_legal_values(csp, var, assignment):
    if csp.domains:
        return len(csp.domains[var])
    else:
        count = 0
        for val in csp.original_domains[var]:
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
    # Prof. Keffer: don't over think this one too much
    vars_to_check = []
    size = []
    for v in csp.variables:  # check in domain?
        if v not in assignment.keys():
            vars_to_check.append(v)
            size.append(num_legal_values(csp, v, assignment))
    # Prof. Keffer: doesn't make if there's multiple same minimums, picking one
    # at random is fine
    # backtracking will handle it
    return vars_to_check[size.index(min(size))]


def order_domain_values(var, assignment, csp):
    """
    The default value order.
    """
    return csp.choices(var)


def mac(csp, var, value, assignment, removals):
    """
    Maintain arc consistency.
    """
    return ac3(csp)


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
        var = minimum_remaining_values(csp, assignment)
        for value in order_domain_values(var, assignment, csp):
            print('Looking at value =', value)
            if ac3(csp):
                print('No conflicts found')
                csp.assign(var, value, assignment)
                # below line is supposed to be inferences in pseudocode
                removals = csp.suppose(var, value)  # inferences is AC3
                if mac(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    else:
                        csp.n_bt += 1
                csp.restore(removals)
        print('No solution, bouncing out of backtrack')
        csp.unassign(var, assignment)
        return None

    result2 = backtrack({})
    print('result2 =', result2)
    assert result2 is None or csp.goal_test(result2)
    return result2


def faulty(csp):
    """
    Check if board just doesn't make sense.
    :param csp:
    :return:
    """
    for x in csp.assignments:
        csp.domains[x] = csp.assignments[x].copy()
    return not ac3(csp)
    return retval


def already_complete(csp: CSP):
    """
    Check if the board is passed in complete
    :param csp:
    :return:
    """
    if len(csp.assignments) != len(csp.domain):
        return False
    for x in csp.assignments:
        if csp.assignments[x][0] not in csp.domain[x]:
            return False
    return ac3(csp)


def backtracking_search_old(csp: CSP):
    domain_copy = csp.domains.copy()
    if faulty(csp):
        return (None, None)
    if already_complete(csp):
        return (csp.assignments, [domain_copy])
    assignments_backups = csp.assignments.copy()
    smallest_val = minimum_remaining_values(csp, csp)
    for i in range(len(domain_copy[smallest_val])):
        possible_candidate = domain_copy[smallest_val][i]
        assignments_backups[smallest_val] = [possible_candidate]

        back_track_progress, domain_list = backtracking_search(csp)
        if back_track_progress is not None:
            domain_list.insert(0, domain_copy)
            return (back_track_progress, domain_list)
    return (None, None)


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
        preview=convert_board_to_html(
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
    a = backtracking_search(SudokuCSP(builtin_puzzles[puzzle_num]))

    return preview_base_temp.substitute(
        preview=convert_board_to_html(
            a
        )
    )


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
