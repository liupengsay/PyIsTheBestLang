"""

"""
"""
算法：回文数字枚举
功能：xxx
题目：
L2081 k 镜像数字的和（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）枚举 k 进制的回文数字并依次判定合法性

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
class PalindromeNum:
    def __init__(self):
        return

    def gen_result(self, n=10):
        # 使用动态规划模拟对称的回文子串添加
        dp = [[""], [str(i) for i in range(n)]]
        for k in range(2, 12):
            if k % 2 == 1:
                m = k // 2
                lst = []
                for st in dp[-1]:
                    for i in range(n):
                        if st[0] != "0":
                            lst.append(st[:m] + str(i) + st[m:])
                dp.append(lst)
            else:
                lst = []
                for st in dp[-2]:
                    for i in range(n):
                        if i != 0:
                            lst.append(str(i) + st + str(i))
                dp.append(lst)
        # 计算出长度小于 n 的所有回文数
        return dp

    @staticmethod
    def get_recent_palindrom_num(n: str) -> list:
        # 564. 寻找最近的回文数（https://leetcode.cn/problems/find-the-closest-palindrome/）
        # P1609 最小回文数（https://www.luogu.com.cn/problem/P1609）
        # 用原数的前半部分加一后的结果替换后半部分得到的回文整数。
        # 用原数的前半部分减一后的结果替换后半部分得到的回文整数。
        # 为防止位数变化导致构造的回文整数错误，因此直接构造 999…999 和 100…001 作为备选答案
        # 计算正整数 n 附近的回文数，获得最近的最小或者最大的回文数

        m = len(n)
        candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
        prefix = int(n[:(m + 1) // 2])
        for x in range(prefix - 1, prefix + 2):
            y = x if m % 2 == 0 else x // 10
            while y:
                x = x * 10 + y % 10
                y //= 10
            candidates.append(x)
        return candidates



class TestGeneral(unittest.TestCase):

    def test_pllindrome_num(self):
        pn = PalindromeNum()
        dp = pn.gen_result()
        assert len(dp[2]) == 9

        n = "44"
        nums = pn.get_recent_palindrom_num(n)
        nums = [num for num in nums if num > int(n)]
        assert min(nums) == 55
        return


if __name__ == '__main__':
    unittest.main()
