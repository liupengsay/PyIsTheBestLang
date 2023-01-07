"""

"""
"""
算法：数位DP
功能：统计满足一定条件的自然数个数，也可以根据字典序大小特点统计一些特定字符串的个数
题目：
L2376 统计特殊整数（https://leetcode.cn/problems/count-special-integers/）计算小于 n 的特殊正整数个数
L0233 数字 1 的个数
L1706 出现 2 的次数
L0600 不含连续 1 的非负整数
L0902 最大为 N 的数字组合
L1012 至少有 1 位重复的数字
L1067 范围内的数字计数
L1397 找到所有好字符串
P1590 失踪的7


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
class DigitalDP:
    def __init__(self):
        return

    @staticmethod
    def count_special_numbers(n: int) -> int:
        s = str(n)

        @lru_cache(None)
        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:  # 可以跳过当前数位
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if mask >> d & 1 == 0:  # d 不在 mask 中
                    res += f(i + 1, mask | (1 << d),
                             is_limit and d == up, True)
            return res
        return f(0, 0, True, False)

    @staticmethod
    def count_digit_num(num, d):
        # 计算1-num内数字d的个数
        @lru_cache(None)
        def dfs(i, cnt, is_limit, is_num):
            if i == n:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, 0, False, False)

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                res += dfs(i + 1, cnt + int(x == d),
                           is_limit and ceil == x, True)
            return res
        s = str(num)
        n = len(s)
        return dfs(0, 0, True, False)

    @staticmethod
    def count_digit_base(num, d):

        # 使用进制计算1-num内不含数字d的个数 1<=d<=9
        s = str(num)
        i = s.find(str(d))
        if i != -1:
            if d:
                s = s[:i] + str(d - 1) + (len(s) - i - 1) * "9"
            else:
                s = s[:i - 1] + str(int(s[i - 1]) - 1) + (len(s) - i - 1) * "9"
            num = int(s)

        lst = []
        while num:
            lst.append(num % 10)
            if d and lst[-1] >= d:
                lst[-1] -= 1
            elif not d and lst[-1] == 0:  # 不含 0 应该怎么算
                num *= 10
                num -= 1
                lst.append(num % 10)
            num //= 10
        lst.reverse()

        ans = 0
        for x in lst:
            ans *= 9
            ans += x
        return ans

    @staticmethod
    def count_digit_base2(num, d):

        # 使用数位DP计算1-num内不含数字d的个数 0<=d<=9
        @lru_cache(None)
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num:  # 可以跳过当前数位
                res = dfs(i + 1, False, False)
            up = int(s[i]) if is_limit else 9
            for x in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if x != d:
                    res += dfs(i + 1, is_limit and x == up, True)
            return res
        s = str(num)
        m = len(s)
        return dfs(0, True, False)

    @staticmethod
    def compute_digit(num, d):
        # 使用进制计算第num个不含数字d的数 0<=d<=9
        lst = []
        st = list(range(10))
        st.remove(d)
        while num:
            if d:
                lst.append(num % 9)
                num //= 9
            else:
                lst.append((num - 1) % 9)
                num = (num - 1) // 9
        lst.reverse()
        # 也可以使用二分加数位DP进行求解
        ans = [str(st[i]) for i in lst]
        return int("".join(ans))


class TestGeneral(unittest.TestCase):

    def test_digital_dp(self):

        dd = DigitalDP()
        cnt = [0] * 10
        n = 1000
        for i in range(1, n + 1):
            for w in str(i):
                cnt[int(w)] += 1

        for d in range(10):
            assert dd.count_digit_num(n, d) == cnt[d]

        for d in range(1, 10):
            assert dd.count_digit_base(
                n, d) == sum(
                str(d) not in str(num) for num in range(
                    1, n + 1))

        for d in range(10):
            assert dd.count_digit_base2(
                n, d) == sum(
                str(d) not in str(num) for num in range(
                    1, n + 1))

        for d in range(10):
            nums = []
            for i in range(1, n + 1):
                if str(d) not in str(i):
                    nums.append(i)
            for i, num in enumerate(nums):
                assert dd.compute_digit(i + 1, d) == num
        return


if __name__ == '__main__':
    unittest.main()
