"""
算法：双指针 快慢指针
功能：通过相对移动来减少计算复杂度
题目：xx（xx）
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


class TwoPointer:
    def __init__(self):
        return

    @staticmethod
    def fast_and_slow(head):
        # 快慢指针判断链表是否存在环
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    @staticmethod
    def same_direction(nums):
        # 相同方向双指针（寻找最长不含重复元素的子序列）
        n = len(nums)
        ans = j = 0
        pre = set()
        for i in range(n):
            # 特别注意指针的移动情况
            while j < n and nums[j] not in pre:
                pre.add(nums[j])
                j += 1
            # 视情况更新返回值
            ans = ans if ans > j - i + 1 else j - i + 1
            pre.discard(nums[i])
        return ans

    @staticmethod
    def opposite_direction(nums, target):
        # 相反方向双指针（寻找升序数组是否存在两个数和为target）
        n = len(nums)
        i, j = 0, n - 1
        while i < j:
            cur = nums[i] + nums[j]
            if cur > target:
                j -= 1
            elif cur < target:
                i += 1
            else:
                return True
        return False


class TestGeneral(unittest.TestCase):

    def test_two_pointer(self):
        nt = TwoPointer()
        nums = [1, 2, 3, 4, 4, 3, 3, 2, 1, 6, 3]
        assert nt.same_direction(nums) == 5

        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert nt.opposite_direction(nums, 9)
        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert not nt.opposite_direction(nums, 16)
        return


if __name__ == '__main__':
    unittest.main()
