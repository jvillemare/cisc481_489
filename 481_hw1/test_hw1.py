import unittest
import hw1

stack9 = [1, 2, 3]


class TestHomeworkOne(unittest.TestCase):
    def test_possible_actions(self):
        self.assertEqual(
            [1, 2, 3],
            hw1.possible_actions([1, 2, 3]),
            'For a stack of 3 pancakes, there is only three possible flips'
        )

    def test_result(self):
        # flip in the middle
        self.assertEqual(
            [2, 1, 3],
            hw1.result(2, stack9),
            'The flip at index 2 should change 123 to 213'
        )
        # flip nothing
        self.assertEqual(
            [1, 2, 3],
            hw1.result(0, stack9),
            'The flip at index 0 should change nothing'
        )
        # flip end
        self.assertEqual(
            [3, 2, 1],
            hw1.result(3, stack9),
            'The flip at index 3 should change 123 to 321'
        )

    def test_expand(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
