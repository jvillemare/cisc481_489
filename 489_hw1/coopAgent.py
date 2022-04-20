from env import *
from typing import List

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
        arr[index] = [arr[index][0] / 50, arr[index][1] / 50]


def generate_heatmap(size: int,
                     coin_values: List[int],
                     coin_positions: List[List[int]],
                     wall_positions: List[List[int]]) -> List[List[int]]:
    """
    Generate a heatmap of
    :param size:
    :param coin_positions:
    :param wall_positions:
    :return:
    """
    pass


def find_nth_peak(heatmap: List[List[int]], n: int) -> List[int, int]:
    """
    Find the n-th highest peak in a heatmap.
    :param heatmap:
    :param n:
    :return:
    """
    pass


def djikstra():
    pass


def convert_path_to_actions(path: List[List[int]]) -> List[str]:
    pass


def pathfind_to_pos(x: int, y: int) -> List[str]:
    pass


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

    def is_player_collide_wall(self):
        for w in walls:
            if self.rect.colliderect(w):
                return True
        return False

    def update(self):
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
        coin_values, coin_positions = get_coin_data()
        wall_positions = get_wall_data()
        correct_positions(coin_positions)
        correct_positions(wall_positions)

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

    def is_player_collide_wall(self):
        for w in walls:
            if self.rect.colliderect(w):
                return True
        return False

    def update(self):

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
