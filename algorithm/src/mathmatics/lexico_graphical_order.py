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
"""
算法：字典序
功能：计算字典序第K小、和某个对象的字典序rank
题目：
440. 字典序的第K小数字（https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/）xxxx
P1243 排序集合（https://www.luogu.com.cn/problem/P1243）求出第K小的子集
P1338 末日的传说（https://www.luogu.com.cn/problem/P1338）结合逆序对计数的字典序
参考：OI WiKi（xx）
"""


class LexicoGraphicalOrder:
    def __init__(self):
        return

    @staticmethod
    def get_kth_num(n, k):

        # 求 [1, n] 范围内字典序第 k 小的数字
        def check():
            c = 0
            first = last = num
            while first <= n:
                c += min(last, n) - first + 1
                last = last * 10 + 9
                first *= 10
            return c

        # assert k <= n
        num = 1
        k -= 1
        while k:
            cnt = check()
            if k >= cnt:
                num += 1
                k -= cnt
            else:
                num *= 10
                k -= 1
        return num

    def get_num_rank(self, n, num):

        # 求 [1, n] 范围内数字 num 的字典序
        x = str(num)
        low = 1
        high = n
        while low < high - 1:
            mid = low + (high - low) // 2
            st = str(self.get_kth_num(n, mid))
            if st < x:
                low = mid
            elif st > x:
                high = mid
            else:
                return mid
        return low if str(self.get_kth_num(n, low)) == x else high

    @staticmethod
    def get_kth_subset(n, k):

        # 集合 [1,..,n] 的第 k 小的子集，总共有 1<<n 个子集
        # assert k <= (1 << n)
        ans = []
        if k == 1:
            # 空子集输出 0
            ans.append(0)
        k -= 1
        for i in range(1, n + 1):
            if k == 0:
                break
            if k <= pow(2, n - i):
                ans.append(i)
                k -= 1
            else:
                k -= pow(2, n - i)
        return ans

    def get_subset_rank(self, n, lst):

        # 集合 [1,..,n] 的子集 lst 的字典序
        low = 1
        high = n
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset(n, low) == lst else high

    @staticmethod
    def get_kth_subset_comb(n, m, k):
        # 集合 [1,..,n] 中选取 m 个元素的第 k 个 comb 选取排列
        # assert k <= math.comb(n, m)

        nums = list(range(1, n + 1))
        ans = []
        while k and nums and len(ans) < m:
            length = len(nums)
            c = math.comb(length - 1, m - len(ans) - 1)
            if c >= k:
                ans.append(nums.pop(0))
            else:
                k -= c
                nums.pop(0)
        return ans

    def get_subset_comb_rank(self, n, m, lst):
        # 集合 [1,..,n] 中选取 m 个元素的排列 lst 的字典序

        low = 1
        high = math.comb(n, m)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_comb(n, m, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_comb(n, m, low) == lst else high

    @staticmethod
    def get_kth_subset_perm(n, k):
        # 集合 [1,..,n] 中选取 n 个元素的第 k 个 perm 选取排列
        # assert k <= math.factorial(n)

        nums = list(range(1, n+1))
        ans = []
        while k and nums:
            single = math.factorial(len(nums)-1)
            i = (k-1) // single
            ans.append(nums.pop(i))
            k -= i*single
        return ans

    def get_subset_perm_rank(self, n, lst):
        # 集合 [1,..,n] 中选取 n 个元素的全排列 lst 的字典序

        low = 1
        high = math.factorial(n)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_perm(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_perm(n, low) == lst else high


class TestGeneral(unittest.TestCase):

    def test_lexico_graphical_order(self):
        lgo = LexicoGraphicalOrder()

        n = 10**5
        nums = sorted([str(x) for x in range(1, n + 1)])
        for _ in range(100):
            i = random.randint(0, n - 1)
            num = nums[i]
            assert lgo.get_kth_num(n, i + 1) == int(num)
            assert lgo.get_num_rank(n, int(num)) == i + 1

        n = 10
        nums = []
        for i in range(1 << n):
            nums.append([j + 1 for j in range(n) if i & (1 << j)])
        nums.sort()
        nums[0] = [0]
        for _ in range(100):
            i = random.randint(0, n - 1)
            lst = nums[i]
            assert lgo.get_kth_subset(n, i + 1) == lst
            assert lgo.get_subset_rank(n, lst) == i + 1

        n = 10
        m = 4
        nums = []
        for item in combinations(list(range(1, n+1)), m):
            nums.append(list(item))
        for _ in range(100):
            i = random.randint(0, len(nums) - 1)
            lst = nums[i]
            assert lgo.get_kth_subset_comb(n, m, i+1) == lst
            assert lgo.get_subset_comb_rank(n, m, lst) == i + 1

        n = 8
        nums = []
        for item in permutations(list(range(1, n+1)), n):
            nums.append(list(item))
        for i, lst in enumerate(nums):
            lst = nums[i]
            assert lgo.get_kth_subset_perm(n, i + 1) == lst
            assert lgo.get_subset_perm_rank(n, lst) == i + 1
        return


if __name__ == '__main__':
    unittest.main()
