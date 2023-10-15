
import unittest
from math import inf


def max_(a, b):
    return a if a > b else b


def min_(a, b):
    return a if a < b else b

def max_lst(args):
    res = -inf
    for x in args:
        if x > res:
            res = x
    return res


def min_lst(args):
    res = inf
    for x in args:
        if x < res:
            res = x
    return res


class TestGeneral(unittest.TestCase):

    def test_max_min_(self):
        for a, b in [[1, 2], [2, 1]]:
            for _ in range(1000000):
                max_(a, b)
                min_(a, b)
        return

    def test_max_min(self):
        for a, b in [[1, 2], [2, 1]]:
            for _ in range(1000000):
                max(a, b)
                min(a, b)
        return

    def test_max_min_lst_(self):
        lst = list(range(100000))
        for _ in range(100):
            max_lst(lst)
            min_lst(lst)
        return

    def test_max_min_lst(self):
        lst = list(range(100000))
        for _ in range(100):
            max(lst)
            min(lst)
        return


if __name__ == '__main__':
    unittest.main()
