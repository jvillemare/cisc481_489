from env import *
from typing import List, Dict, Set, Tuple
import numpy as np
import random

# Design your own agent(s) in this file.
# You can use your own favorite icon or as simple as a colored square (with
# different colors) to represent your agent(s).
playerA_img = pygame.image.load(os.path.join("img", "playerA.png")).convert()
playerB_img = pygame.image.load(os.path.join("img", "playerB.png")).convert()


def correct_positions(arr: List[int]) -> None:
    """
    Correct the positions in the coin data and wall positions by dividing them
    by 50 to get their "true" position.
    :param arr: Operation is performed on array, no result is returned
    :return: None
    """
    for index in range(len(arr)):
        arr[index] = [int(arr[index][0] / 50), int(arr[index][1] / 50)]


HEATMAP_COLDEST = 0
HEATMAP_AMBIENT = 300  # base temperature
HEATMAP_HOTTEST = 5000


def normalize_coin(raw: int) -> int:
    """
    Max-min normalize a coin value into temperature.
    :param raw: Coin value.
    :return: Scaled temperature value of coin.
    """
    return int(((raw - 1) / (9 - 1)) * HEATMAP_HOTTEST)


def generate_heatmap(size: int,
                     coin_values: List[int],
                     coin_positions: List[List[int]],
                     wall_positions: List[List[int]]) -> np.ndarray:
    """
    Generate a heatmap of board.
    :param size: The length of the board, (size x size)
    :param coin_values: The values of the coins
    :param coin_positions: The positions of the coins. Assuming these values
    have been normalized.
    :param wall_positions: The position of the walls. Assuming these values
    have been normalized.
    :return: A heat map.
    """
    # make n x n matrix set to coldest temp
    heatmap = np.full((size, size), HEATMAP_AMBIENT)
    heatmap_avg = np.empty((size, size))
    # apply the heat of the coin values
    for x in range(len(coin_values)):
        heatmap[coin_positions[x][1]][coin_positions[x][0]] = \
            normalize_coin(coin_values[x])
    # apply the cold of the walls
    for w in wall_positions:
        heatmap[w[1]][w[0]] = HEATMAP_COLDEST
    # bleed the temps, average it out
    r = 3  # radius
    for x in range(size):
        for y in range(size):
            heatmap_avg[x][y] = \
                round(
                    np.mean(
                        heatmap[
                            max(x-r, 0):min(x+r, size),
                            max(y-r, 0):min(y+r, size)
                        ]
                    ),
                    2
                )
    print('=' * 80)
    print('Heatmap:')
    pretty_print_2d(heatmap_avg)
    return heatmap_avg


def pretty_print_2d(arr) -> None:
    """Print a 2D array with tabs in between."""
    print('\n'.join(['\t\t'.join([str(cell) for cell in row]) for row in arr]))


def find_peak_index(a: np.ndarray, blacklist: List[List[int]]) -> List[int]:
    """
    Find the index of the largest value in a numpy ndarray not in a blacklist.
    :param a: Numpy array to search in.
    :param blacklist: Indexes to exclude when searching for peak.
    :return: Index of non-blacklist peak.
    """
    peak, peak_index = -1000, None
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            if a[x][y] > peak and [x, y] not in blacklist:
                peak = a[x][y]
                peak_index = [x, y]
    assert peak_index is not None, 'This should not be happening'
    return peak_index
    # return np.unravel_index(a.argmax(), a.shape)


class Dijkstra:
    """
    Dijkstra's algorithm.
    Special thanks to https://stackoverflow.com/a/61078380/13158722
    I have modified the code from the stackoverflow answer to slightly suit my
    needs here.
    """

    def __init__(self, vertices, graph):
        self.vertices = vertices  # ("A", "B", "C" ...)
        self.graph = graph  # {"A": {"B": 1}, "B": {"A": 3, "C": 5} ...}

    def find_route(self, start, end):
        unvisited = {n: float("inf") for n in self.vertices}
        unvisited[start] = 0  # set start vertex to 0
        visited = {}  # list of all visited nodes
        parents = {}  # predecessors
        while unvisited:
            # get the smallest distance
            min_vertex = min(unvisited, key=unvisited.get)
            for neighbour, _ in self.graph.get(min_vertex, {}).items():
                if neighbour in visited:
                    continue
                new_distance = unvisited[min_vertex] + self.graph[min_vertex].get(neighbour, float("inf"))
                # Prof. Keffer: check with BFS, because here we do not have
                # uniform path cost values here.
                if new_distance < unvisited[neighbour]:
                    unvisited[neighbour] = new_distance
                    parents[neighbour] = min_vertex
            visited[min_vertex] = unvisited[min_vertex]
            unvisited.pop(min_vertex)
            if min_vertex == end:
                break
        return parents, visited

    @staticmethod
    def generate_path(parents, start, end):
        """
        Calclualte path by parents from start to end.
        :param parents: As calculated by find_route method in this class.
        :param start: Begin vertex.
        :param end: End vertex.
        :return: The path of vertices, or None if no path is possible.
        """
        print('Generating path with start =', start, 'and end =', end)
        path = [end]
        try:
            while True:
                key = parents[path[0]]
                path.insert(0, key)
                if key == start:
                    break
        except KeyError:
            print('No path is possible')
            return None
        return path


def convert_str_to_pos(s: str) -> List[int]:
    """Convert a string representation of a cell to a list of int coordinates"""
    return list(map(int, s.replace('C', '').split(',')))


def convert_pos_to_str(pos: List[int]) -> str:
    """Convert a list of int coordinates of a cell to a string representation"""
    return 'C' + ','.join(list(map(str, pos)))


def generate_graph(agent: int,
                   size: int,
                   coin_values: List[int],
                   coin_positions: List[List[int]],
                   wall_positions: List[List[int]],
                   exclude_pos: List[int]) -> \
        Tuple[Set[str], Dict[str, None]]:
    """
    Generates the list of edges used in Djikstra from a list of wall positions.
    :param agent: The agent that is running this function.
    :param size: The length of the square board (size x size).
    :param coin_values: The values of the coins.
    :param coin_positions: The positions of the coins.
    :param wall_positions: A list containing two element lists of normalized
    wall positions. E.g. `[ [3, 2], ... ]
    :param exclude_pos: A position to exclude, including its adjacent positions.
    :return: A tuple containing the vertices and graph.
    """
    print('Generating graph for Agent', agent)
    print('Exclude position passed to generate_graph is', exclude_pos)
    vertices = set()
    graph = {}
    # exclude neighboring positions of exclude position
    ex_left =  [exclude_pos[0] - 1, exclude_pos[1]]
    ex_right = [exclude_pos[0] + 1, exclude_pos[1]]
    ex_up =    [exclude_pos[0],     exclude_pos[1] - 1]
    ex_down =  [exclude_pos[0],     exclude_pos[1] + 1]

    # make vertices
    for x in range(size + 1):
        for y in range(size + 1):
            pos = [x, y]
            # ignore the exclude position
            if pos == exclude_pos or pos == ex_left or pos == ex_right or pos\
                    == ex_up or pos == ex_down:
                continue
            # exclude all walls, they're not traversable
            if pos not in wall_positions:
                vertices.add('C' + str(x) + ',' + str(y))

    # make graph
    def bad(pos):
        """Check if a position is out of bounds or in a wall"""
        return pos[0] < 0 or pos[0] > size or pos[1] < 0 or pos[1] > size or \
               pos in wall_positions or pos == exclude_pos or pos == ex_left \
               or pos == ex_right or pos == ex_up or pos == ex_down

    def weighted_val(pos):
        """Calculate the weighted value of a position"""
        return 100 - (coin_values[coin_positions.index(pos)] * 10) if pos in \
                                                              coin_positions \
            else 100

    for v in vertices:
        center = convert_str_to_pos(v)
        neighbors = {}

        # I had to do this explicitly instead of my nice for loop because:
        # 1. the double nested for loop was causing a ton of problems
        # 2. there's only 4 max neighbors a cell can have
        # 3. this solution SHOULD be fool proof
        # 4. the graph looks better (more reasonable) when I changed to this
        left =  [center[0] - 1, center[1]]
        right = [center[0] + 1, center[1]]
        up =    [center[0],     center[1] - 1]
        down =  [center[0],     center[1] + 1]
        # i indented the code this way because i made a mistake here and didn't
        # notice it for the longest time; headache

        if not bad(left):
            neighbors[convert_pos_to_str(left)] = weighted_val(left)
        if not bad(right):
            neighbors[convert_pos_to_str(right)] = weighted_val(right)
        if not bad(up):
            neighbors[convert_pos_to_str(up)] = weighted_val(up)
        if not bad(down):
            neighbors[convert_pos_to_str(down)] = weighted_val(down)

        # this pos was causing so many problems. I feel like as it is in its
        # code block right now, it SHOULD work. but for some damn reason it
        # keeps failing. no idea. keeping it here for prosperity.
        """
        for x in range(center[0] - 1, center[0] + 2):
            for y in range(center[1] - 1, center[1] + 2):
                # don't go out of the board
                if x < 0 or x > size or y < 0 or y > size:
                    continue
                # don't include walls or itself
                if [x, y] in wall_positions or [x, y] == v:
                    continue
                # don't include diagonals
                
                # Weight every node as 10...
                value = 10
                if [x, y] in coin_positions:
                    # But make it better based on the value, so we can swing by
                    # and grab coins on the way if they are good
                    coin_idx = coin_positions.index([x, y])
                    value -= coin_values[coin_idx]
                neighbors[convert_pos_to_str([x, y])] = value
        """
        graph[v] = neighbors
    return vertices, graph


def path_find(agent: int,
              start: List[int],
              end: List[int],
              size: int,
              coin_values: List[int],
              coin_positions: List[List[int]],
              wall_positions: List[List[int]],
              exclude_pos: List[int]) -> List[str]:
    """
    Find a path between a start and end position, for a board of a given size,
    and given the positions of the walls.
    :param agent: The agent that is calling this function.
    :param start: Two element list, like `[x, y]`.
    :param end: Two element list, like `[x, y]`.
    :param size: The length of the board, like (size x size).
    :param coin_values: The values of the coins.
    :param coin_positions: The position of the coins.
    :param wall_positions: List of two element lists containing the positions of
    the walls, assuming they have been "normalized".
    :param exclude_pos: A position to exclude in path finding, including its
    adjacent positions. Default is None, and if filled out, should be [x, y]
    :return: The path to take to reach a specified location. Returns None if the
    start and end match.
    """
    if start == end:
        return None
    vertices, graph = generate_graph(agent, size, coin_values, coin_positions,
                                     wall_positions, exclude_pos)
    # TODO: remove prints later
    print('vertices', vertices)
    print('graph', graph)
    d = Dijkstra(vertices, graph)
    start_vertex = convert_pos_to_str(start)
    end_vertex = convert_pos_to_str(end)
    p, v = d.find_route(
        start_vertex,
        end_vertex
    )
    se = d.generate_path(p, start_vertex, end_vertex)
    return se


def convert_path_to_actions(path: List[str]) -> List[str]:
    """
    Converts a path to a list of step actions that can be taken.
    :param path: The path calculated by Djikstra's algorithm.
    :return: The series of move actions ('u', 'r', 'd', 'l') in a list.
    """
    if path is None:
        return None
    actions = []
    # the first element in the path is the current position
    # skip it, and let's figure out the next step
    last = convert_str_to_pos(path[0])
    for p in path[1:]:
        next_step = convert_str_to_pos(p)
        if last[0] + 1 == next_step[0]:  # if x has increased, move right
            actions.append('r')
        elif last[1] + 1 == next_step[1]:  # if y has increased, move down
            actions.append('d')
        elif last[0] - 1 == next_step[0]:  # if x has decreased, move left
            actions.append('l')
        elif last[1] - 1 == next_step[1]:  # if y has decreased, move up
            actions.append('u')
        else:
            raise Exception('This should not be possible, last is', last,
                            'and next_step is', next_step)
        last = convert_str_to_pos(p)
    return actions


def find_nearest_coin(current_pos: List[int],
                      coin_values: List[int],
                      coin_positions: List[List[int]]) -> List[int]:
    """
    Find the position of the nearest coin, weighted. This function is
    weighted by coin values. This means that the coin distance is calculated,
    and then reduced by its value.
    :param current_pos: The current position of the agent.
    :param coin_values: The values of the coins.
    :param coin_positions: The position of the coins.
    :return: The nearest coin.
    """
    distances = []
    pos = np.array(current_pos)
    for idx in range(len(coin_values)):
        distances.append(
            np.linalg.norm(
                pos - np.array(coin_positions[idx])
            ) * (2.0 / coin_values[idx])  # 2.0 is an arbitrary number
        )
    return coin_positions[np.unravel_index(pos.argmin(), pos.shape)[0]]


def construct_initial_plan(agent: int,
                           size: int,
                           wall_positions: List[List[int]],
                           direction: str,
                           paces: int) -> List[str]:
    """
    Construct an initial plan for agents to space out with.
    :param agent: The number of the agent (A = 1, B = 2).
    :param size: Size of the board.
    :param wall_positions: The positions of the walls to avoid.
    :param direction: The direction that they will head for a number of paces.
    :param paces: The number of paces to move.
    :return: The actions needed to space out.
    """
    plan = None

    dest = None
    if direction == 'r':
        dest = [paces, 0]
    elif direction == 'd':
        dest = [0, paces]
    else:
        raise Exception('Invalid direction', direction, 'was passed')

    path = path_find(
        agent,
        [0, 0],
        dest,
        size,
        [],
        [[]],
        wall_positions,
        [-2, -2]
    )
    plan = convert_path_to_actions(path)

    print('Initial plan for Agent', agent, 'is', plan)
    return plan


def reconsider_surroundings(agent: int,
                            position: List[int],
                            coin_values: List[int],
                            coin_positions: List[List[int]],
                            wall_positions: List[List[int]],
                            other_agent_pos: List[int]):
    """
    Reconsider current plan if there is a coin immediately adjacent OR
    diagonal to the agents current position.
    :param agent:
    :param position:
    :param coin_values:
    :param coin_positions:
    :param wall_positions:
    :param other_agent_pos:
    :return: A new list of actions to take to the nearest coin. Otherwise
    returns None if there are no adjacent coins
    """
    pass


def random_action(agent: int,
                  position: List[int],
                  size: int,
                  wall_positions: List[List[int]],
                  other_agent_pos: List[int]) -> str:
    """
    Come up with a random filler action.
    :param agent: The number of the current agent.
    :param position: The position of the current agent.
    :param size: The size of the board.
    :param wall_positions: The positions of the walls on the board.
    :param other_agent_pos: The position of the other agent.
    :return: A random move. None if there are no good moves.
    """
    print('Agent', agent, 'had to resort to a random action, this is not good')
    left =  [position[0] - 1, position[1]]
    right = [position[0] + 1, position[1]]
    up =    [position[0],     position[1] - 1]
    down =  [position[0],     position[1] + 1]

    def bad(pos):
        """Determine if a given position is bad in this context"""
        return pos[0] < 0 or pos[0] > size or pos[1] < 0 or pos[1] > size in \
               wall_positions or pos == other_agent_pos

    if not bad(left):
        return 'l'
    if not bad(right):
        return 'r'
    if not bad(up):
        return 'u'
    if not bad(down):
        return 'd'
    else:
        return None


def action_spelled_out(action: str) -> None:
    """
    Converts a one letter action to fully spelled out ('r' becomes 'right')
    :param action: The one letter action.
    :return: It fully spelled out.
    """
    translate = {
        'r': 'right',
        'l': 'left',
        'd': 'down',
        'u': 'up'
    }
    return translate[action]


def other_agent_adjacent_old(agent: int,
                         pos: List[int],
                         other_pos: List[int],
                         wall_positions: List[List[int]]) -> str:
    """
    DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED.

    If there is another agent adjacent OR DIAGONAL, then come up with a move
    that moves in the opposite direction of the agent.

    This was necessary because I was having too many accidental collisions
    with agents, even with all of my other safety code that I thought would
    handle this.

    :param agent: The ID of the agent calling this function.
    :param pos: The position of the calling agent.
    :param other_pos: The position of the other agent.
    :param wall_positions: The position of the walls.
    :return: None if there is no agent nearby, or a move in a single string
    ('l', 'r', 'u', or 'd') if there is an agent nearby.

    DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED. DEPRECATED.
    """
    # this means that the two agents are one or two moves away from each other
    if np.linalg.norm( np.array(pos) - np.array(other_pos) ) < 2.0:
        a_m = None

        left =  [pos[0] - 1, pos[1]]
        right = [pos[0] + 1, pos[1]]
        up =    [pos[0],     pos[1] - 1]
        down =  [pos[0],     pos[1] + 1]

        # verified by white board drawings
        top_left =     [pos[0] - 1, pos[1] - 1]
        top_right =    [pos[0] + 1, pos[1] - 1]
        bottom_left =  [pos[0] - 1, pos[1] + 1]
        bottom_right = [pos[0] + 1, pos[1] + 1]

        # this is super verbose, and it's because i'm tired of making elegant
        # solutions that fail, and then i just have to resort to hard coding it
        if left == other_pos:
            a_m = 'r'  # go opposite of where other agent is
        if right == other_pos:
            a_m = 'l'
        if up == other_pos:
            a_m = 'd'
        if down == other_pos:
            a_m = 'u'

        # these next returns are going to be somewhat arbitrary, may change
        # later
        if top_left == other_pos:
            a_m = 'e'

        # NOTE: yeah no i'm abandoning this
        # ======================================================================

        print('Agent', agent, 'is too close to another agent, avoid move is',
              action_spelled_out(a_m))
        return a_m
    else:
        return None


def other_agent_adjacent(agent: int,
                         pos1: List[int],
                         pos2: List[int]) -> bool:
    """
    Determines if a agent is adjacent to another agent.
    :param agent: The ID of the agent.
    :param pos1: Position of the first agent.
    :param pos2: Position af the second agent
    :return: True if pos1 is adjacent to pos2, False if not.
    """
    if np.linalg.norm( np.array(pos1) - np.array(pos2) ) < 2.0:
        print('Agent', agent, 'is adjacent to another agent, pos1 =', pos1,
              'pos2 =', pos2)
        return True
    else:
        return False


def opposite_corner(size: int,
                    pos: List[int]) -> List[int]:
    """
    When provided a position, this functions finds the closest corner, then
    returns the corner opposite of it.

    This is used when two agents are close together, just start heading away
    from each other.
    :param size: The size of the board.
    :param pos: The position of the other agent.
    :return: The opposite corner.
    """
    # setup numpy arrays
    top_left =     np.array([0, 0])
    top_right =    np.array([size, 0])
    bottom_left =  np.array([0, size])
    bottom_right = np.array([size, size])
    np_pos = np.array(pos)

    # calculate distances
    tl_d = np.linalg.norm( np.array(top_left) - np.array(np_pos) )
    tr_d = np.linalg.norm(np.array(top_right) - np.array(np_pos))
    bl_d = np.linalg.norm(np.array(bottom_left) - np.array(np_pos))
    br_d = np.linalg.norm(np.array(bottom_right) - np.array(np_pos))

    # top_left has the closest distance, return the opposite corner
    if tl_d < tr_d < bl_d < br_d:
        return list(bottom_right)
    elif tr_d < tl_d < bl_d < br_d:  # top_right closest
        return list(bottom_left)
    elif bl_d < tl_d < tr_d < br_d:  # bottom_left closest
        return list(top_right)
    elif br_d < tl_d < tr_d < bl_d:  # bottom_right closest
        return list(top_left)
    else:  # all equally close?
        return list(random.choice([
            top_left,
            top_right,
            bottom_left,
            bottom_right
        ]))


broadcast_player_a_pos = [0, 0]
broadcast_player_b_pos = [0, 0]


class PlayerA(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = playerA_img
        self.image = pygame.transform.scale(playerA_img, (WALLSIZE, WALLSIZE))
        self.image.set_colorkey(BLACK)
        pygame.draw.rect(
            self.image,
            rand_color(random.randint(0, N)),
            self.image.get_rect(),
            3
        )
        self.rect = self.image.get_rect()  # get image position
        self.rect.x = 0
        self.rect.y = 0
        self.true_x = 0  # I added this to better keep track of my real position
        self.true_y = 0  # I added this to better keep track of my real position
        self.mode = 'START_MODE'  # 'START_MODE', 'HOT_AREA', or 'FOCUS_COIN'
        self.focus_coin = None
        wall_positions = get_wall_data()
        correct_positions(wall_positions)
        self.actions = construct_initial_plan(
            1,
            N,
            wall_positions,
            'r',
            7
        )
        self.look_around_count = 0
        self.speedx = SPEED
        self.speedy = SPEED
        self.score = 0
        self.steps = 0

    def move(self, direction):
        if direction == 'r':
            self.true_x += 1
            self.steps += 1
            self.rect.x += self.speedx
            if self.is_player_collide_wall():
                self.rect.x -= self.speedx
        if direction == 'l':
            self.true_x -= 1
            self.steps += 1
            self.rect.x -= self.speedx
            if self.is_player_collide_wall():
                self.rect.x += self.speedx
        if direction == 'u':
            self.true_y -= 1
            self.steps += 1
            self.rect.y -= self.speedy
            if self.is_player_collide_wall():
                self.rect.y += self.speedy
        if direction == 'd':
            self.true_y += 1
            self.steps += 1
            self.rect.y += self.speedy
            if self.is_player_collide_wall():
                self.rect.y -= self.speedy
        global broadcast_player_a_pos
        broadcast_player_a_pos = [self.true_x, self.true_y]

    def is_player_collide_wall(self):
        for w in walls:
            if self.rect.colliderect(w):
                return True
        return False

    def update(self):  # PLAYER >> A << UPDATE
        """
        # TODO: Please implement how your agents decide to move
        # Based on the information of the Coins and Walls
        # get current time
        print("Current Time in milliseconds:", pygame.time.get_ticks())
        # get current information of the coins
        print("Coin Data:", get_coin_data())
        # get current information of the walls
        print("Wall Positions:", get_wall_data())
        direction = 1
        # print(direction)
        if direction == 0:
            self.move('l')  # move left
        if direction == 1:
            self.move('r')  # move right
        if direction == 2:
            self.move('u')  # move up
        if direction == 3:
            self.move('d')  # move down
        """
        # get all the data
        coin_values, coin_positions = get_coin_data()
        wall_positions = get_wall_data()
        correct_positions(coin_positions)
        correct_positions(wall_positions)
        print('coin_positions', coin_positions)
        print('wall_positions', wall_positions)

        print('Agent 1 is currently at pos', self.rect.x, self.rect.y)
        print('Agent 1 is currently at true pos', self.true_x, self.true_y)

        current_pos = [self.true_x, self.true_y]

        """
        # find the hottest area in the map, and calculate a path to it
        if len(self.actions) == 0: # and self.look_around_count == 0:
            print('Agent 1 finding new hot area...')
            hm = generate_heatmap(N, coin_values, coin_positions, wall_positions)
            peak_x, peak_y = find_peak_index(hm, wall_positions)
            print('Agent 1 moving to hot area', [peak_x, peak_y])
            print('Agent 1 determining path from', current_pos, 'to',
                  [peak_x, peak_y])
            path = path_find(
                1,
                current_pos,
                [peak_x, peak_y],
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            if path == None:
                self.actions = [random_action(
                    1,
                    current_pos,
                    N,
                    wall_positions,
                    broadcast_player_b_pos
                )]
            else:
                self.actions = convert_path_to_actions(path)
            # self.look_around_count = 2

        # if already in the hottest area, look for the nearest coin
        if self.actions is None:
            print('Agent 1 searching for nearest coin...')
            nearest_coin = find_nearest_coin(
                current_pos,
                coin_values,
                coin_positions
            )
            path = path_find(
                1,
                current_pos,
                nearest_coin,
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            print('Agent 1 going to nearest coin:', nearest_coin)
            self.actions = convert_path_to_actions(path)
            # self.look_around_count -= 1

        print('Agent 1 following a plan of length', str(len(self.actions)))
        next_action = self.actions.pop(0)

        # double check for collision
        if other_agent_adjacent(1, broadcast_player_a_pos,
                                broadcast_player_b_pos):
            print('Agent 1 about to collide, adopting plan to run away')
            oc = opposite_corner(N, broadcast_player_b_pos)
            path = path_find(
                1,
                current_pos,
                oc,
                N,
                coin_values,
                coin_positions,
                wall_positions,
                [-2, -2]
            )
            self.actions = convert_path_to_actions(path)
            next_action = self.actions.pop(0)

        """

        """
        if self.mode == 'START_MODE':
            next_action = self.actions.pop(0)
            self.move(next_action)
            if len(self.actions) == 0:
                self.mode = 'HOT_AREA'
        elif self.mode == 'HOT_AREA':
            # reconsider path to hot area
            print('Agent 1 finding new hot area...')
            hm = generate_heatmap(N, coin_values, coin_positions, wall_positions)
            peak_x, peak_y = find_peak_index(hm, wall_positions)
            print('Agent 1 moving to hot area', [peak_x, peak_y])
            print('Agent 1 determining path from', current_pos, 'to', [peak_x,
                                                                           peak_y])
            path = path_find(
                1,
                current_pos,
                [peak_x, peak_y],
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            if path is None:
                self.mode == 'FOCUS_COIN'
            else:
                self.actions = convert_path_to_actions(path)
                next_action = self.actions.pop(0)
                self.move(next_action)
        if self.mode == 'FOCUS_COIN':
            if self.focus_coin is None:
                nearest_coin = find_nearest_coin(
                    current_pos,
                    coin_values,
                    coin_positions
                )
                print('Agent 1 pos', broadcast_player_a_pos)
                print('Agent 2 pos', broadcast_player_b_pos)
                path = path_find(
                    2,
                    current_pos,
                    nearest_coin,
                    N,
                    coin_values,
                    coin_positions,
                    wall_positions,
                    broadcast_player_a_pos
                )
                if path is None:
                    whitelisted_coins_pos = coin_positions.copy()
                    whitelisted_coins_vals = coin_values.copy()
                    # since the nearest coin is unreachable, remove it from the list
                    remove_index = coin_positions.index(nearest_coin)
                    whitelisted_coins_pos.pop(remove_index)
                    whitelisted_coins_vals.pop(remove_index)
                    next_nearest_coin = None
                    while next_nearest_coin is None:
                        next_nearest_coin = find_nearest_coin(
                            current_pos,
                            [1] * len(whitelisted_coins_pos),
                            whitelisted_coins_pos
                        )
                        path = path_find(
                            2,
                            current_pos,
                            next_nearest_coin,
                            N,
                            coin_values,
                            whitelisted_coins_pos,
                            wall_positions,
                            broadcast_player_a_pos
                        )
                        if path is None:
                            remove_index = coin_positions.index(next_nearest_coin)
                            whitelisted_coins_pos.pop(remove_index)
                            whitelisted_coins_vals.pop(remove_index)
                            next_nearest_coin = None
                    nearest_coin = next_nearest_coin

                self.focus_coin = nearest_coin
                print('Agent 1 going to nearest coin:', nearest_coin)
                self.actions = convert_path_to_actions(path)
            if self.focus_coin is not None:
                print('Agent 1 following a plan of length',
                      str(len(self.actions)))
                next_action = self.actions.pop(0)
                self.move(next_action)
        """

        # just keep looking for the nearest coin, and go for it
        if len(self.actions) == 0:
            nearest_coin = find_nearest_coin(
                [self.true_x, self.true_y],
                coin_values,
                coin_positions
            )
            print('Agent 1 pos', broadcast_player_a_pos)
            print('Agent 2 pos', broadcast_player_b_pos)
            path = path_find(
                2,
                [self.true_x, self.true_y],
                nearest_coin,
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            if path is None:
                whitelisted_coins_pos = coin_positions.copy()
                whitelisted_coins_vals = coin_values.copy()
                # since the nearest coin is unreachable, remove it from the list
                remove_index = coin_positions.index(nearest_coin)
                whitelisted_coins_pos.pop(remove_index)
                whitelisted_coins_vals.pop(remove_index)
                next_nearest_coin = None
                while next_nearest_coin is None:
                    next_nearest_coin = find_nearest_coin(
                        [self.true_x, self.true_y],
                        [1] * len(whitelisted_coins_pos),
                        whitelisted_coins_pos
                    )
                    path = path_find(
                        2,
                        [self.true_x, self.true_y],
                        next_nearest_coin,
                        N,
                        coin_values,
                        whitelisted_coins_pos,
                        wall_positions,
                        broadcast_player_b_pos
                    )
                    if path is None:
                        remove_index = whitelisted_coins_pos.index(next_nearest_coin)
                        whitelisted_coins_pos.pop(remove_index)
                        whitelisted_coins_vals.pop(remove_index)
                        next_nearest_coin = None

            print('Agent 1 going to nearest coin:', nearest_coin)
            self.actions = convert_path_to_actions(path)

        print('Agent 1 following a plan of length', str(len(self.actions)))
        next_action = self.actions.pop(0)
        self.move(next_action)

        # print('=' * 80)
        # h = generate_heatmap(N, coin_values, coin_positions, wall_positions)
        # pretty_print_2d(h)

        # Avoid colliding with wall and go out of edges
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
        if self.rect.top < 0:
            self.rect.top = 0


# You can design another player class to represent the other player if they work
# in different ways.
class PlayerB(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = playerB_img
        self.image = pygame.transform.scale(playerB_img, (WALLSIZE, WALLSIZE))
        self.image.set_colorkey(BLACK)
        pygame.draw.rect(self.image, rand_color(random.randint(0, N)), self.image.get_rect(), 3)
        self.rect = self.image.get_rect()  # get image position
        self.rect.x = 0
        self.rect.y = 0
        self.true_x = 0  # I added this to better keep track of my real position
        self.true_y = 0  # I added this to better keep track of my real position
        wall_positions = get_wall_data()
        correct_positions(wall_positions)
        self.actions = construct_initial_plan(
            1,
            N,
            wall_positions,
            'd',
            5
        )
        self.speedx = SPEED
        self.speedy = SPEED
        self.score = 0
        self.steps = 0

    def move(self, direction):
        if direction == 'r':
            self.true_x += 1
            self.steps += 1
            self.rect.x += self.speedx
            if self.is_player_collide_wall():
                self.rect.x -= self.speedx
        if direction == 'l':
            self.true_x -= 1
            self.steps += 1
            self.rect.x -= self.speedx
            if self.is_player_collide_wall():
                self.rect.x += self.speedx
        if direction == 'u':
            self.true_y -= 1
            self.steps += 1
            self.rect.y -= self.speedy
            if self.is_player_collide_wall():
                self.rect.y += self.speedy
        if direction == 'd':
            self.true_y += 1
            self.steps += 1
            self.rect.y += self.speedy
            if self.is_player_collide_wall():
                self.rect.y -= self.speedy
        global broadcast_player_b_pos
        broadcast_player_b_pos = [self.true_x, self.true_y]

    def is_player_collide_wall(self):
        for w in walls:
            if self.rect.colliderect(w):
                return True
        return False

    def update(self):  # PLAYER >> B << UPDATE
        coin_values, coin_positions = get_coin_data()
        wall_positions = get_wall_data()
        correct_positions(coin_positions)
        correct_positions(wall_positions)

        # just keep looking for the nearest coin, and go for it
        if len(self.actions) == 0:
            nearest_coin = find_nearest_coin(
                [self.true_x, self.true_y],
                [1] * len(coin_positions),
                coin_positions
            )
            print('Agent 1 pos', broadcast_player_a_pos)
            print('Agent 2 pos', broadcast_player_b_pos)
            path = path_find(
                2,
                [self.true_x, self.true_y],
                nearest_coin,
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_a_pos
            )
            if path is None:
                whitelisted_coins_pos = coin_positions.copy()
                whitelisted_coins_vals = coin_values.copy()
                # since the nearest coin is unreachable, remove it from the list
                remove_index = coin_positions.index(nearest_coin)
                whitelisted_coins_pos.pop(remove_index)
                whitelisted_coins_vals.pop(remove_index)
                next_nearest_coin = None
                while next_nearest_coin is None:
                    next_nearest_coin = find_nearest_coin(
                        [self.true_x, self.true_y],
                        [1] * len(whitelisted_coins_pos),
                        whitelisted_coins_pos
                    )
                    path = path_find(
                        2,
                        [self.true_x, self.true_y],
                        next_nearest_coin,
                        N,
                        coin_values,
                        whitelisted_coins_pos,
                        wall_positions,
                        broadcast_player_a_pos
                    )
                    if path is None:
                        remove_index = whitelisted_coins_pos.index(next_nearest_coin)
                        whitelisted_coins_pos.pop(remove_index)
                        whitelisted_coins_vals.pop(remove_index)
                        next_nearest_coin = None

            print('Agent 2 going to nearest coin:', nearest_coin)
            self.actions = convert_path_to_actions(path)

        print('Agent 2 following a plan of length', str(len(self.actions)))
        next_action = self.actions.pop(0)
        self.move(next_action)


        """
        direction = 3
        # print(direction)
        if direction == 0:  # left
            self.move('l')
        if direction == 1:  # right
            self.move('r')
        if direction == 2:  # up
            self.move('u')
        if direction == 3:  # down
            self.move('d')
        """

        # Avoid colliding with wall and go out of edges
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
        if self.rect.top < 0:
            self.rect.top = 0

# Hint: To cooperate, it's better if your agents explore different areas of the
# map, so you can write a communication function to broadcast their locations
# in order that they can keep a reasonable distance from each other.
# The bottom line is at least they shouldn't collide with each other.
# You may try different strategies (e.g. reactive, heuristic, learning, etc).
