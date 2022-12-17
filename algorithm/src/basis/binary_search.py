"""


"""
"""
算法：二分查找、有序集合
功能：利用单调性确定最优选择，通常可以使用SortedList
题目：xx（xx）
L2468 根据限制分割消息（https://leetcode.cn/problems/split-message-based-on-limit/）根据长度限制进行二分
L2426 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
L2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
L2141 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
L2102 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询
参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class DefineSortedList:
    def __init__(self, lst=[]):
        # 也可以考虑使用bisect实现
        self.lst = lst
        self.lst.sort()
        return

    def add(self, num):
        if not self.lst:
            self.lst.append(num)
            return
        if self.lst[-1] <= num:
            self.lst.append(num)
            return
        if self.lst[0] >= num:
            self.lst.insert(0, num)
            return
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            self.lst.insert(i, num)
        else:
            self.lst.insert(j, num)
        return

    def discard(self, num):
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] > num:
                j = mid
            elif self.lst[mid] < num:
                i = mid
            else:
                self.lst.pop(mid)
                return
        if self.lst[i] == num:
            self.lst.pop(i)
        elif self.lst[j] == num:
            self.lst.pop(j)
        return

    def bisect_left(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] < num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            return i
        return j

    def bisect_right(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] <= num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0

        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] <= num:
                i = mid
            else:
                j = mid
        if self.lst[j] >= num:
            return j
        return i


class TestGeneral(unittest.TestCase):

    def test_define_sorted_list(self):
        for _ in range(10):
            floor = -10**8
            ceil = 10**8
            low = -5*10**7
            high = 6*10**8
            n = 10**4
            # add
            lst = SortedList()
            define = DefineSortedList([])
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
            assert all(lst[i] == define.lst[i] for i in range(n))
            # discard
            for _ in range(n):
                num = random.randint(low, high)
                lst.discard(num)
                define.discard(num)
            m = len(lst)
            assert all(lst[i] == define.lst[i] for i in range(m))
            # bisect_left
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_left(num) == define.bisect_left(num)
            # bisect_right
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_right(num) == define.bisect_right(num)
        return


if __name__ == '__main__':
    unittest.main()

