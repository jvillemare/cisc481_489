from typing import List, Any


def minus_lists(a: List[List[Any]], b: List[List[Any]]) -> list:
    """
    From list a, subtract the items from list b. Returning a brand new list
    object.

    Amazingly, I can't find a solution for this online, so I have to built my
    own implementation of this.

    The
    :param a: List to subtract from.
    :param b: List to subtract with.
    :return: Brand new list, a - b.
    """
    new_a = a.copy()
    for b_i in b:
        if b_i in new_a:
            del new_a[new_a.index(b_i)]
    return new_a


if __name__ == '__main__':
    a = [[1, 2], [3, 4]]
    b = [[3, 4], [9, 9], [], None]

    c = minus_lists(a, b)

    print('a', a)
    print('b', b)
    print('c', c)
