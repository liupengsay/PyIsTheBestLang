"""

"""

"""
算法：SPFA：路径边数优先的广度优先搜索（可以使用带负权值）

功能：SPFA（Shortest Path Faster Algorithm）是一种用于计算单源最短路径的算法。它通过使用队列和松弛操作来不断更新路径长度，从而更快地找到最短路径。

下面是一个简单的 Python SPFA 模板，其中 graph 是图的邻接表表示，src 是源节点，dist 是各节点到源节点的最短距离，prev 是各节点的前驱节点。
上面的代码只是一个简单的 SPFA 模板，实际使用时可能需要添加更多的特判和优化。例如，SPFA 算法在某些情况下容易陷入死循环，因此需要添加防止死循环的机制。此外，SPFA 算法的时间复杂度与输入图

的稠密程度有关，因此可能需要使用一些优化方法来提高它的效率。

功能：SPFA 算法是一种简单易用的最短路径算法，它通过使用队列和松弛操作来快速求解单源最短路径问题。它的时间复杂度与输入图的稠密程度有关，并且容易陷入死循环，因此需要注意这些问题。
Dijkstra：路径权值优先的深度优先搜索（只适用正权值）

参考题目：
P3385 负环（https://www.luogu.com.cn/problem/P3385）通过最短路径更新的边数来计算从起点出发是否存在负环
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
import heapq
import sys
from collections import defaultdict, Counter, deque
from functools import lru_cache


class SPFA:
    def __init__(self, dct):
        self.dct = dct
        self.n = len(dct)
        # 初始化距离
        self.dis = [float("inf") for _ in range(self.n)]
        # 标识当前节点是否在栈中
        self.visit = [False] * self.n
        # 当前最小距离的路径边数
        self.cnt = [0] * self.n
        return

    def negtive_circle(self):
        # 求带负权的最短路距离与路径边数
        queue = deque([0])
        # 队列与起点初始化默认从 0 出发
        self.dis[0] = 0
        self.visit[0] = True

        while queue:
            # 取出队列中的第一个节点
            u = queue.popleft()
            self.visit[u] = False
            # 更新当前节点的相邻节点的距离
            for v, w in self.dct[u].items():
                if self.dis[v] > self.dis[u] + w:
                    self.dis[v] = self.dis[u] + w
                    self.cnt[v] = self.cnt[u] + 1
                    if self.cnt[v] >= self.n:
                        return "YES"
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not self.visit[v]:
                        queue.append(v)
                        self.visit[v] = True
        # 不存在从起点出发的负环
        return "NO"


class TestGeneral(unittest.TestCase):

    def test_spfa(self):
        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {}]
        spfa = SPFA(dct)
        res = spfa.negtive_circle()
        assert res == "NO"
        assert spfa.dis == [0, 5, 1, 3]
        assert spfa.cnt == [0, 1, 1, 2]

        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {2: -4}]
        spfa = SPFA(dct)
        res = spfa.negtive_circle()
        assert res == "YES"
        return


if __name__ == '__main__':
    unittest.main()
