
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
算法：分治、递归、二叉树、四叉树、十叉树、N叉树
功能：xxx
题目：
P1911 L 国的战斗之排兵布阵（https://www.luogu.com.cn/problem/P1911）使用四叉树递归计算
P5461 赦免战俘（https://www.luogu.com.cn/problem/P5461）递归计算四叉树左上角

参考：OI WiKi（xx）
"""



class Recusion:
    def __init__(self):
        return

    def gen_result(self):


        return

    @staticmethod
    def main_p1911(n, x, y):

        x -= 1
        y -= 1
        m = 1 << n
        ans = [[0] * m for _ in range(m)]

        def dfs(x1, y1, x2, y2, a, b):
            nonlocal ind
            if x1 == x2 and y1 == y2:
                return
            flag = find(x1, y1, x2, y2, a, b)
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2

            # # 四叉树中心邻居节点
            lst = [[x0, y0], [x0, y0 + 1], [x0 + 1, y0], [x0 + 1, y0 + 1]]
            nex = []
            for i in range(4):
                if i != flag:
                    ans[lst[i][0]][lst[i][1]] = ind
                    nex.append(lst[i])
                else:
                    nex.append([a, b])
            ind += 1
            # 四叉树递归坐标
            dfs(x1, y1, x0, y0, nex[0][0], nex[0][1])
            dfs(x1, y0 + 1, x0, y2, nex[1][0], nex[1][1])
            dfs(x0 + 1, y1, x2, y0, nex[2][0], nex[2][1])
            dfs(x0 + 1, y0 + 1, x2, y2, nex[3][0], nex[3][1])
            return

        def find(x1, y1, x2, y2, a, b):
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2
            if x1 <= a <= x0 and y1 <= b <= y0:
                return 0
            if x1 <= a <= x0 and y0 + 1 <= b <= y2:
                return 1
            if x0 + 1 <= a <= x2 and y1 <= b <= y0:
                return 2
            return 3

        ind = 1

        dfs(0, 0, m - 1, m - 1, x, y)
        dct = dict()
        dct[0] = 0
        for i in range(m):
            for j in range(m):
                x = ans[i][j]
                if x not in dct:
                    dct[x] = len(dct)

        return [[dct[i] for i in a] for a in ans]


class TestGeneral(unittest.TestCase):

    def test_rescursion(self):
        # nt = ClassName()
        # assert nt.gen_result(1 0* *11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
