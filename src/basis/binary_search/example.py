import unittest

from src.basis.binary_search.template import BinarySearch


class TestGeneral(unittest.TestCase):

    def test_binary_search(self):
        bs = BinarySearch()

        def check(xx):
            nonlocal tm
            tm += 1
            return xx >= y

        for x in range(1, 7):
            n = 10**x
            lst = []
            for y in range(1, n + 1):
                tm = 0
                bs.find_int_left_strictly(1, n, check)
                lst.append(tm)
            assert (1 << max(lst)) >= n > (1 << (max(lst) - 1))

        def check_right(xx):
            nonlocal tm
            tm += 1
            return xx <= y

        for x in range(1, 7):
            n = 10**x
            lst = []
            for y in range(1, n + 1):
                tm = 0
                bs.find_int_right_strictly(1, n, check_right)
                lst.append(tm)
            assert (1 << max(lst)) >= n > (1 << (max(lst) - 1))
        return


if __name__ == '__main__':
    unittest.main()
