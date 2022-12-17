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

    def gen_result(self):
        # 使用动态规划模拟对称的回文子串添加
        dp = [[""], [str(i) for i in range(10)]]
        for k in range(2, 12):
            if k % 2 == 1:
                m = k // 2
                lst = []
                for st in dp[-1]:
                    for i in range(10):
                        if st[0] != "0":
                            lst.append(st[:m] + str(i) + st[m:])
                dp.append(lst)
            else:
                lst = []
                for st in dp[-2]:
                    for i in range(10):
                        if i != 0:
                            lst.append(str(i) + st + str(i))
                dp.append(lst)
        return dp


class TestGeneral(unittest.TestCase):

    def test_pllindrome_num(self):
        pn = PalindromeNum()
        dp = pn.gen_result()
        assert len(dp[2]) == 9
        return


if __name__ == '__main__':
    unittest.main()
