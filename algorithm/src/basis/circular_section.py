
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
算法：循环节
功能：通过模拟找出循环节进行状态计算
题目：
Lxxxx xxxx（https://leetcode.cn/problems/shortest-palindrome/）xxxx
P1965 [NOIP2013 提高组] 转圈游戏（https://www.luogu.com.cn/problem/P1965）循环节计算
957. N 天后的牢房（https://leetcode.cn/problems/prison-cells-after-n-days/）循环节计算
418. 屏幕可显示句子的数量（https://leetcode.cn/problems/sentence-screen-fitting/）循环节计算
466. 统计重复个数（https://leetcode.cn/problems/count-the-repetitions/）循环节计算
P1532 卡布列克圆舞曲（https://www.luogu.com.cn/problem/P1532）循环节计算
P2203 Blink（https://www.luogu.com.cn/problem/P2203）循环节计算
参考：OI WiKi（xx）
"""


class CircleSection:
    def __init__(self):
        return

    @staticmethod
    def compute_circle_result(n, m, x, tm):  # 循环节模板

        # 计算 x 每次加 m 加了 m 次后模 n 的循环态
        dct = dict()
        lst = []
        while x not in dct:
            dct[x] = len(lst)
            lst.append(x)
            x = (x + m) % n

        # 此时加 m 次数状态为 0.1...length-1
        length = len(lst)
        # 在 ind 次处出现循节
        ind = dct[x]

        # 所求次数不超出循环节
        if tm < length:
            return lst[tm]

        # 所求次数进入循环节
        circle = length - ind
        tm -= length
        j = tm % circle
        return lst[ind + j]


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        # nt = ClassName()
        # assert nt.gen_result(1 0* *11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
