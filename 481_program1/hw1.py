"""
hw1.py
The Pancake Sorting Problem
This is a variation on the general idea of sorting. The idea is you have a stack
of pancakes, each one a different size from all the others. Initially, the
pancakes are stacked in some random order, and we’d like to get them stacked in
order of size from smallest on top to largest on the bottom. At any point in
time, you can flip a portion of the stack by placing a spatula in between two
pancakes and turning over, as a block, the pancakes above the spatula.
---
@date       March 15, 2022
@author     James Villemarette
"""

# imports
from memory_profiler import profile         # for profiling the functions
from typing import List                     # for type hinting
import sys                                  # for bypassing recursion limit

sys.setrecursionlimit(1500)

# stacks from the program1-writeup.pdf
stack0 = [1, 3, 5, 2, 4, 6]                 # solved stack
stack1 = [8, 2, 1, 7, 5, 4, 6, 3, 9, 10]    # solvable in 7 flips
stack2 = [9, 5, 2, 8, 4, 1, 10, 6, 7, 3]
stack3 = [2, 8, 10, 5, 7, 3, 4, 6, 1, 9]
stack4 = [3, 6, 8, 10, 7, 1, 5, 4, 2, 9]
stack5 = [6, 9, 4, 8, 1, 3, 2, 7, 10, 5]
stack6 = [8, 5, 10, 6, 2, 9, 3, 4, 1, 7]
stack7 = [8, 1, 10, 5, 3, 7, 4, 9, 2, 6]
stack8 = [1, 2, 3]

stack0_goal = [1, 2, 3, 4, 5, 6]
stack1_7_goals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

stacks1_7 = {
    'stack1': stack1,
    'stack2': stack2,
    'stack3': stack3,
    'stack4': stack4,
    'stack5': stack5,
    'stack6': stack6,
    'stack7': stack7
}


class Pancake:
    """
    A pancake.
    """

    stack: List[int]
    """
    List of ints, where the value refers to the size of the pancake in the 
    stack.
    """

    action: str
    """
    The action applied to make this stack.
    """

    parent = None
    """
    The stack before this one.
    """

    cost: int
    """
    How long it took to get here.
    """

    depth: int
    """
    How far we made it.
    """

    """
    Hold all the important shiz.
    """
    def __init__(self, stack, action, parent):
        self.stack = stack
        self.action = action
        self.parent = parent
        self.cost = 0

        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0


# helper functions


def pretty_pancake_print(stack: List[int], actions: List[int]) -> None:
    """
    Takes in a stack list and a list of actions, and then pretty prints out the
    stack and the list of actions that apply to it.
    :param stack: The stack of pancakes.
    :param actions: Actions, where every int is like [(flip 2), (flip 5), ...]
    :return: Nothing.
    """
    print('Original Stack', stack)
    cur_stack = stack.copy()
    for act in actions:
        print('Flip', act)
        cur_stack = result(act, cur_stack)
        print('New Stack', cur_stack)


# Part 1 - Possible Actions


#@profile
def possible_actions(stack: List[int]) -> List[int]:
    """
    Takes a stack as input and outputs a list of all actions possible on the 
    given stack.
    :param stack: The stack of pancakes.
    :return: List of all actions.
    """
    actions = list(range(2, len(stack) + 1))
    return actions


# Part 2 - Result


#@profile
def result(action: int, stack: List[int]) -> List[int]:
    """
    Takes as input an action and a stack and outputs the new stack that will
    result after actually carrying out the input move in the input state.
    Be certain that you do not accidentally modify the input stack variable
    :param action: Action, where every int is like [(flip 2), (flip 5), ...]
    :param stack: The stack of pancakes.
    :return: A new list where the flip is applied.
    """
    new_stack = list(reversed( stack[0:action] )) + stack[action:]
    return new_stack

# Part 3 - Expand


#@profile
def expand(state: Pancake) -> List[Pancake]:
    """
    Takes a stack as input, and outputs a list of all states that can be reached
    in one Action from the given state.
    :param state: The state containing the stack of pancakes.
    :return: A list, containing lists that are the states that can be reached.
    """
    pa = possible_actions(state.stack)
    possible_states = []
    for act in pa:
        if act != state.action:  # don't do the same thing twice
            possible_states.append(
                Pancake(result(act, state.stack), act, state)
            )
    return possible_states

# Part 4 - IDS


@profile
def iterative_deepening_search(initial_stack: List[int], goal_stack: List[int]) -> List[int]:
    """
    Takes an initial stack and a goal stack and produces a list of actions that
    form an optimal path from the initial stack to the goal.
    :param initial_stack: The base stack of pancakes.
    :param goal_stack: The goal stack of pancakes.
    :return: List of actions.
    """
    temp_state = Pancake(initial_stack, None, None)
    pancake_lists = expand(temp_state)
    max_depth = len(initial_stack)
    optimal = temp_state

    checked = []
    keep_looking = True
    while keep_looking:
        for stack in pancake_lists:
            # no double checking, no too deep search
            if (stack not in checked) and (stack.depth < max_depth):
                checked.append(stack)
                temp_state = stack
                if temp_state.stack == goal_stack:
                    optimal = temp_state
                    max_depth = temp_state.depth
                    # break
                if temp_state.depth < max_depth:
                    pancake_lists = expand(temp_state) + pancake_lists
                    break
                print('len(pancake_lists) =', len(pancake_lists))
                keep_looking = False

    good_actions = []
    while optimal.parent is not None:
        good_actions.insert(0, optimal.action)
        optimal = optimal.parent

    return good_actions
    """
    for cur_depth in range(0, sys.maxsize):
        dls_result = depth_limited_search(initial_stack, goal_stack, cur_depth)
        if dls_result == goal_stack:
            return dls_result
    """
    # ==========================================================================
    # for i in range(25):
    #     if depth_limited_search(initial_stack, goal_stack, i):
    #         return True
    # return False
    # ==========================================================================
    # for cur_depth in range(0, sys.maxsize):
    #     dls_result = depth_limited_search(initial_stack, goal_stack, cur_depth)
    #     if dls_result == goal_stack:
    #         return dls_result


@profile
def depth_limited_search(initial_stack: List[int], goal_stack: List[int], depth: int):
    """
    No longer used. This was going to be for an approach on iterative
    deepening search that turnedo ut to be fruitless.
    :param initial_stack:
    :param goal_stack:
    :param depth:
    :return:
    """
    frontier = [initial_stack]
    if depth % 100 == 0:
        print('len(frontier): ', len(frontier), ', goal: ', goal_stack, ', depth: ', depth)
    depth_count = 0
    while len(frontier) > 0:
        print(frontier)
        node = frontier.pop()
        #print('node: ', node, ', goal: ', goal_stack, ', depth: ', depth)
        if goal_stack == node:
            return node
        if depth_count > depth:
            return False
        if node not in frontier:
            for i in expand(node):
                #print('adding new node to frontier', i)
                frontier.append(i)
        depth_count += 1
    # ==========================================================================
    # if initial_stack == goal_stack:
    #     return True
    #
    # # If reached the maximum depth, stop recursing.
    # if max_depth <= 0:
    #     return False
    #
    # # Recur for all the vertices adjacent to this vertex
    # for s in initial_stack:
    #     if depth_limited_search(s, goal_stack, max_depth - 1):
    #         return True
    # return False
    # ==========================================================================
    # frontier = [initial_stack]
    # dls_result = False
    # while len(frontier) > 0:
    #     node = frontier.pop()
    #     if goal_stack == node:
    #         return node
    #     if


# Part 5 - BFS


@profile
def breadth_first_search(initial_stack: List[int], goal_stack: List[int]) -> List[int]:
    """
    takes an initial stack and a goal and gives an optimal sequence of actions
    from the initial state to the goal.
    :param initial_stack:
    :param goal_stack:
    :return:
    """
    temp_state = Pancake(initial_stack, None, None)
    pancake_lists = [temp_state]
    checked = []
    while temp_state.stack != goal_stack:
        search_next = []  # stop double searches
        for pancake_looper in pancake_lists:
            if pancake_looper.stack not in checked:
                temp_state = pancake_looper
                if temp_state.stack == goal_stack:
                    print('pancake_lists =', len(pancake_lists))
                    break
                more_temp_state = expand(temp_state)
                search_next = search_next + more_temp_state
                checked.append(pancake_looper.stack)
        pancake_lists = list(set(search_next)) # remove duplicated
    ze_actions = []
    while temp_state.parent is not None:
        ze_actions.insert(0, temp_state.action)
        temp_state = temp_state.parent
    return ze_actions

# Part 6 - A* Search


def a_star_heuristic(pancake):
    """
    Calculate the heuristic cost of A* search.
    :param pancake:
    :return:
    """
    if pancake.parent is not None:
        pancake.cost += pancake.parent.cost
    i = 0
    while i < len(pancake.stack) - 2:
        if abs(pancake.stack[i] - pancake.stack[i + 1]) != 1:
            pancake.cost += 1
        i += 1
    return pancake


def sort_pancake_cost(p: Pancake) -> int:
    """Sort function, just return the cost."""
    return p.cost


@profile
def a_star_search(initial_stack: List[int], goal_stack: List[int]) -> List[int]:
    """
    Similar to breadth-first and iterative deepening, your A* search should take
    as input an initial stack and a goal stack. It should additionally take a
    heuristic function

    You can test your code by passing as the heuristic a function that always
    returns, which should reduce your search to uniform-cost search. Test it on
    *stack-0*. You can also try it on *stack-1* for good measure. It may take a
    long time to solve any of the other.
    :param initial_stack:
    :param goal_stack:
    :return:
    """
    temp_state = a_star_heuristic(Pancake(initial_stack, None, None))
    pancake_lists = []
    while temp_state.stack != goal_stack:
        temp_list = []
        for pancake in expand(temp_state):
            temp_list.append(a_star_heuristic(pancake))

        if temp_state.depth < len(initial_stack):
            pancake_lists = pancake_lists + temp_list
            pancake_lists.sort(key=sort_pancake_cost)

            temp_state = pancake_lists[0]
            pancake_lists.pop(0)
        else:
            temp_state = pancake_lists[0]
    print('pancake_lists =', len(pancake_lists))
    ze_actions = []
    while temp_state.parent is not None:
        ze_actions.insert(0, temp_state.action)
        temp_state = temp_state.parent
    return ze_actions


@profile
def main():
    print('=' * 80)
    print('iterative_deepening_search')
    solution = iterative_deepening_search(stack0, stack0_goal)
    print('stack0 =', stack0, 'stack0_goal =', stack0_goal, 'solution =',
        solution)
    """
    for s in stacks1_7:
        solution = iterative_deepening_search(stacks1_7[s], stack1_7_goals)
        print(
            'stack =', s,
            'initial_stack =',
            stacks1_7[s],
            'goal_stack =',
            stack1_7_goals,
            'solution =',
            solution
        )
    """
    print('=' * 80)
    print('breadth_first_search')
    solution = breadth_first_search(stack0, stack0_goal)
    print('stack0 =', stack0, 'stack0_goal =', stack0_goal, 'solution =',
          solution)
    # solution = breadth_first_search(stack1, stack1_7_goals)
    # print('stack1 =', stack1, 'stack1_goal =', stack1_7_goals, 'solution =',
    #      solution)
    print('=' * 80)
    print('a_star_search')
    solution = a_star_search(stack0, stack0_goal)
    print('stack0 =', stack0, 'stack0_goal =', stack0_goal, 'solution =',
          solution)


if __name__ == '__main__':
    main()
