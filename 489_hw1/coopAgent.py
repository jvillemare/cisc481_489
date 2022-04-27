from env import *
from typing import List, Dict, Set, Tuple
import numpy as np

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


def find_peak_index(a: np.ndarray, blacklist: List[List[int]]) -> Tuple:
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
        path = [end]
        while True:
            key = parents[path[0]]
            path.insert(0, key)
            if key == start:
                break
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
        return 40 - (coin_values[coin_positions.index(pos)] * 4) if pos in \
                                                              coin_positions \
            else 40

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
        self.actions = ['r', 'r', 'r', 'r', 'r']   # I added this to carry a
        # path
        # plan to
        # future states
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

        # find the hottest area in the map, and calculate a path to it
        if len(self.actions) == 0:
            print('Agent 1 finding new hot area...')
            hm = generate_heatmap(N, coin_values, coin_positions, wall_positions)
            peak_x, peak_y = find_peak_index(hm, wall_positions)
            print('Agent 1 moving to hot area', [peak_x, peak_y])
            print('Agent 1 determining path from', [self.true_x,
                                                    self.true_y], 'to',
                  [peak_x, peak_y])
            path = path_find(
                1,
                [self.true_x, self.true_y],
                [peak_x, peak_y],
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            self.actions = convert_path_to_actions(path)

        # if already in the hottest area, look for the nearest coin
        if self.actions is None:
            print('Agent 1 searching for nearest coin...')
            nearest_coin = find_nearest_coin(
                [self.true_x, self.true_y],
                coin_values,
                coin_positions
            )
            path = path_find(
                1,
                [self.true_x, self.true_y],
                nearest_coin,
                N,
                coin_values,
                coin_positions,
                wall_positions,
                broadcast_player_b_pos
            )
            print('Agent 1 going to nearest coin:', nearest_coin)
            self.actions = convert_path_to_actions(path)

        print('Agent 1 following a plan of length', str(len(self.actions)))
        next_action = self.actions.pop(0)
        self.move(next_action)

        #print('=' * 80)
        #h = generate_heatmap(N, coin_values, coin_positions, wall_positions)
        #pretty_print_2d(h)

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
        self.actions = ['d', 'd', 'd', 'd', 'd']
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
