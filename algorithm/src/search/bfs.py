"""

"""
"""
算法：广度优先搜索
功能：在有向图与无向图进行扩散，多源双向BFS，0-1BFS（类似SPFA）
题目：
L2493 将节点分成尽可能多的组（https://leetcode.cn/problems/shortest-palindrome/）利用并查集和广度优先搜索进行连通块分组并枚举最佳方案
L2290 到达角落需要移除障碍物的最小数（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）使用0-1 BFS进行优化计算最小代价
L1368 使网格图至少有一条有效路径的最小代价（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用0-1 BFS进行优化计算最小代价
L2258 逃离火灾（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用二分查找加双源BFS进行模拟
P5507 机关（https://www.luogu.com.cn/problem/P5507）双向BFS进行搜索
L2092 找出知晓秘密的所有专家（https://leetcode.cn/problems/find-all-people-with-secret/）按照时间排序，在同一时间进行BFS扩散
P1747 好奇怪的游戏（https://www.luogu.com.cn/problem/P1747）双向BFS搜索最短距离
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
class BFS:
    def __init__(self):
        return

    @staticmethod
    def bilateral_bfs():
        # 双向BFS
        # P1747 好奇怪的游戏
        state = []
        dct = [[0] * 4 for _ in range(12)]
        for i in range(12):
            nums = [int(x) for x in input().strip().split() if x]
            for j in range(1, 5):
                dct[i][j - 1] = nums[j] - 1
            state.append(nums[0] - 1)

        # 定义两个搜索状态
        stack1 = [tuple(state)]
        visit1 = {stack1[0]: tuple()}
        target = tuple([0] * 12)
        flag = (target == stack1[0][0])
        step = 1

        stack2 = [target]
        visit2 = {stack2[0]: tuple()}
        while not flag:
            nex1 = []
            for pre in stack1:
                lst = list(pre)
                for i in range(12):
                    tmp = lst[:]
                    x = dct[i][tmp[i]]
                    tmp[x] += 1
                    tmp[x] %= 4
                    tmp[i] += 1
                    tmp[i] %= 4
                    tmp = tuple(tmp)
                    if tmp not in visit1:
                        nex1.append(tmp)
                        visit1[tmp] = visit1[pre] + (i + 1,)
                        # 判定是否遇见
                        if tmp in visit2:
                            flag = True
                            path = visit1[tmp] + visit2[tmp][::-1]
                            break
                if flag:
                    break

            nex2 = []
            for pre in stack2:
                lst = list(pre)
                for i in range(12):
                    tmp = lst[:]

                    tmp[i] -= 1
                    tmp[i] %= 4

                    x = dct[i][tmp[i]]
                    tmp[x] -= 1
                    tmp[x] %= 4

                    tmp = tuple(tmp)
                    if tmp not in visit2:
                        nex2.append(tmp)
                        visit2[tmp] = visit2[pre] + (i + 1,)
                        # 判定是否遇见
                        if tmp in visit1:
                            flag = True
                            path = visit1[tmp] + visit2[tmp][::-1]
                            break
                if flag:
                    break
            stack1 = nex1
            stack2 = nex2
            step += 1
        print(len(path))
        print(" ".join(str(x) for x in path))
        return

    @staticmethod
    def main_p1747(x0, y0, x2, y2):

        # 双向BFS模板题

        def check(x1, y1):
            if (x1, y1) == (1, 1):
                return 0

            visit1 = {(x1, y1): 0}
            visit2 = {(1, 1): 0}
            direc = [[1, 2], [1, -2], [-1, 2], [-1, -2],
                     [2, 1], [2, -1], [-2, 1], [-2, -1]]
            direc.extend([[2, 2], [2, -2], [-2, 2], [-2, -2]])
            stack1 = [[x1, y1]]
            stack2 = [[1, 1]]
            step = 1

            while True:
                nex1 = []
                for i, j in stack1:
                    for a, b in direc:
                        if 0 < i + a <= 20 and 0 < j + b <= 20 and (i + a, j + b) not in visit1:
                            visit1[(i + a, j + b)] = step
                            nex1.append([i + a, j + b])
                            if (i + a, j + b) in visit2:
                                return step + visit2[(i + a, j + b)]

                stack1 = nex1

                nex2 = []
                for i, j in stack2:
                    for a, b in direc:
                        if 0 < i + a <= 20 and 0 < j + b <= 20 and (i + a, j + b) not in visit2:
                            visit2[(i + a, j + b)] = step
                            nex2.append([i + a, j + b])
                            if (i + a, j + b) in visit1:
                                return step + visit1[(i + a, j + b)]

                stack2 = nex2
                step += 1
            return -1

        ans1 = check(x0, y0)
        ans2 = check(x2, y2)
        return ans1, ans2


class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        # L2290
        m, n = len(grid), len(grid[0])
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        q = deque([(0, 0)])
        while q:
            x, y = q.popleft()
            for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= nx < m and 0 <= ny < n:
                    g = grid[x][y]
                    if dis[x][y] + g < dis[nx][ny]:
                        dis[nx][ny] = dis[x][y] + g
                        if g == 0:
                            q.appendleft((nx, ny))
                        else:
                            q.append((nx, ny))
        return dis[m - 1][n - 1]


class Solution:
    # L1368
    def minCost(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        BIG = int(1e9)
        dist = [0] + [BIG] * (m * n - 1)
        seen = set()
        q = collections.deque([(0, 0)])

        while len(q) > 0:
            x, y = q.popleft()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            cur_pos = x * n + y
            for i, (nx, ny) in enumerate(
                    [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]):
                new_pos = nx * n + ny
                new_dis = dist[cur_pos] + (1 if grid[x][y] != i + 1 else 0)
                if 0 <= nx < m and 0 <= ny < n and new_dis < dist[new_pos]:
                    dist[new_pos] = new_dis
                    if grid[x][y] == i + 1:
                        q.appendleft((nx, ny))
                    else:
                        q.append((nx, ny))

        return dist[m * n - 1]


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
