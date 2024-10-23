import random
import unittest

from src.structure.priority_queue.template import HeapqMedian, FindMedian
from src.structure.sorted_list.template import SortedList


class TestGeneral(unittest.TestCase):

    def test_heapq_median(self):
        ceil = 10000
        lst = SortedList()
        hm = FindMedian()
        for i in range(ceil):
            num = random.randint(0, ceil)*2
            x = random.randint(0, 5)
            if x == 0 and lst:
                i = random.randint(0, len(lst) - 1)
                hm.remove(lst.pop(i))
            else:
                lst.add(num)
                hm.add(num)
            if not lst:
                continue
            assert len(lst) == hm.small_cnt + hm.big_cnt
            if len(lst) % 2:
                assert lst[len(lst)//2] == hm.find_median()
            else:
                assert lst[len(lst) // 2] +  lst[len(lst) // 2 - 1] == hm.find_median()*2
        return


if __name__ == '__main__':
    unittest.main()
