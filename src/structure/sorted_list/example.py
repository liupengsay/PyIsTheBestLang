import random
import unittest

from src.structure.sorted_list.template import SortedList


class TestGeneral(unittest.TestCase):

    def test_custom_sorted_list(self):

        for _ in range(10):
            floor = -10 ** 8
            ceil = 10 ** 8
            low = -5 * 10 ** 7
            high = 6 * 10 ** 8
            n = 10 ** 4
            # add
            lst = SortedList()
            local_lst = SortedList()
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                local_lst.add(num)
            assert all(lst[i] == local_lst[i] for i in range(n))
            # discard
            for _ in range(n):
                num = random.randint(low, high)
                lst.discard(num)

                local_lst.discard(num)
            m = len(lst)
            assert all(lst[i] == local_lst[i] for i in range(m))
            # bisect_left
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                local_lst.add(num)
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_left(num) == local_lst.bisect_left(num)
            # bisect_right
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_right(num) == local_lst.bisect_right(num)
        return


if __name__ == '__main__':
    unittest.main()
