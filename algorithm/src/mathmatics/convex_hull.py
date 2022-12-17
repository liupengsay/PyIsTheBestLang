"""

"""
"""
算法：凸包
功能：求点集的子集组成最小凸包上
题目：
L1924 安装栅栏 II（https://leetcode.cn/problems/erect-the-fence-ii/）求出最小凸包后使用三分套三分求解最小圆覆盖

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


class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[float]:
        # 1.求凸包
        # 1.1 找到最小最左的点作为凸包扫描的起点
        point = [float('inf'), float('inf')]
        for tree in trees:
            if [tree[1], tree[0]] < point:
                point = tree[:]

        # 1.2 计算所有点到起点的极角与距离
        def angle(node):
            x, y = node
            cur = math.atan2(y - point[1], x - point[0]) * 180 / math.pi
            return [cur, abs(x - point[0])]

        n = len(trees)
        for i in range(n):
            trees[i].extend(angle(trees[i]))
        trees.sort(key=lambda x: [x[2], x[3]])

        # 1.3 最小凸包生成对于共线的点只保留较远的端点
        def check(node1, node2, node3):  # 判断是否逆时针
            x2, y2 = node2[0] - node1[0], node2[1] - node1[1]
            x3, y3 = node3[0] - node1[0], node3[1] - node1[1]
            return x2 * y3 - y2 * x3 < 0

        stack = []
        for node in trees:
            if not node[2]:
                if len(stack) == 2:
                    stack.pop(-1)
            else:
                if stack and node[2] == stack[-1][2]:
                    stack.pop(-1)
                else:
                    while len(stack) >= 3 and check(stack[-2], stack[-1], node):
                        stack.pop(-1)
            stack.append(node)
        del trees

        if len(stack) == 1:
            return point + [0]

        # 2.三分套三分搜寻极值
        def target(x, y):
            return max([(x - p[0]) ** 2 + (y - p[1]) ** 2 for p in stack])

        eps = 5e-8
        lowx = min([p[0] for p in stack])
        highx = max([p[0] for p in stack])
        lowy = min([p[1] for p in stack])
        highy = max([p[1] for p in stack])

        def optimize(y):
            low = lowx
            high = highx
            while low < high - eps:
                diff = (high - low) / 3
                mid1 = low + diff
                mid2 = low + 2 * diff
                dist1 = target(mid1, y)
                dist2 = target(mid2, y)
                if dist1 < dist2:
                    high = mid2
                elif dist1 > dist2:
                    low = mid1
                else:
                    low = mid1
                    high = mid2
            return low, target(low, y)

        low = lowy
        high = highy
        while low < high - eps:
            diff = (high - low) / 3
            mid1 = low + diff
            mid2 = low + 2 * diff
            _, dist1 = optimize(mid1)
            _, dist2 = optimize(mid2)
            if dist1 < dist2:
                high = mid2
            elif dist1 > dist2:
                low = mid1
            else:
                low = mid1
                high = mid2
        x, r = optimize(low)
        return [x, low, math.sqrt(r)]


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()