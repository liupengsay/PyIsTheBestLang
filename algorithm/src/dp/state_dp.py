"""

"""
"""
算法：状态压缩DP
功能：使用二进制数字表示转移状态，计算相应的转移方程，通常可以先计算满足条件的子集，有时通过深搜回溯枚举全部子集的办法比位运算枚举效率更高
题目：

===================================力扣===================================
1723. 完成所有工作的最短时间（https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/）通过位运算枚举分配工作DP最小化的最大值
1986. 完成任务的最少工作时间段（https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）预处理计算子集后进行记忆化状态转移
698. 划分为k个相等的子集（https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/）预处理计算子集后进行记忆化状态转移
1349. 参加考试的最大学生数（https://leetcode.cn/problems/maximum-students-taking-exam/）按行状态枚举所有的摆放可能性
2172. 数组的最大与和（https://leetcode.cn/problems/maximum-and-sum-of-array/）使用位运算和状态压缩进行转移

===================================洛谷===================================
P2704 炮兵阵地（https://www.luogu.com.cn/problem/P2704）记录两个前序状态进行转移
P1896 互不侵犯（https://www.luogu.com.cn/problem/P1896）按行状态与行个数枚举所有的摆放可能性
P2196 [NOIP1996 提高组] 挖地雷（https://www.luogu.com.cn/problem/P2196）有向图最长路径加状压DP
P1690 贪婪的Copy（https://www.luogu.com.cn/problem/P1690）最短路加状压DP
P1294 高手去散步（https://www.luogu.com.cn/problem/P1294）图问题使用状压DP求解最长直径
P1123 取数游戏（https://www.luogu.com.cn/problem/P1123）类似占座位的经典状压DP
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


class StateDP:
    def __init__(self):
        return

    @staticmethod
    def main_l1349(seats: List[List[str]]) -> int:
        lst = []
        for se in seats:
            st = "".join(["0" if x == "." else "1" for x in se])
            lst.append(int("0b" + st, 2))

        @lru_cache(None)
        def dfs(state, i):
            if i >= m:
                return 0
            if i < m - 1:
                res = dfs(lst[i + 1], i + 1)
            else:
                res = 0
            ind = [j for j in range(n) if not state & (1 << j)]
            for k in range(1, len(ind) + 1):
                for item in combinations(ind, k):
                    if all(item[x + 1] - item[x] > 1 for x in range(k - 1)):
                        if i < m - 1:
                            sta = lst[i + 1]
                            for x in item:
                                for y in [x - 1, x + 1]:
                                    if 0 <= y < n:
                                        sta |= (1 << y)
                            nex = k + dfs(sta, i + 1)
                        else:
                            nex = k
                        res = res if res > nex else nex
            return res

        m = len(seats)
        n = len(seats[0])
        return dfs(lst[0], 0)

    @staticmethod
    def main_p1896(n, k):

        @lru_cache(None)
        def dfs(state, i, x):
            # [上一行状态，当前行索引，已有国王个数]
            if x == k:
                return 1
            if i == n or x > k:
                return 0

            # 当前行不摆放
            res = dfs(0, i + 1, x)

            # 当前行可摆放的位置
            ind = []
            for j in range(n):
                if all(not state & (1 << w)
                       for w in [j - 1, j, j + 1] if 0 <= w < n):
                    ind.append(j)

            # 枚举当前行摆放方案
            m = len(ind)
            for length in range(1, m + 1):
                if x + length <= k:
                    for item in combinations(ind, length):
                        if any(item[j + 1] - item[j] ==
                               1 for j in range(length - 1)):
                            continue
                        nex = sum(1 << w for w in item)
                        res += dfs(nex, i + 1, x + length)
            return res

        return dfs(0, 0, 0)


class TestGeneral(unittest.TestCase):

    def test_state_dp(self):
        sd = StateDP()
        assert sd.main_p1896(9, 12) == 50734210126

        seats = [["#", ".", ".", ".", "#"], [".", "#", ".", "#", "."], [".", ".", "#", ".", "."], [".", "#", ".", "#", "."], ["#", ".", ".", ".", "#"]]
        assert sd.main_l1349(seats) == 10

        return


if __name__ == '__main__':
    unittest.main()
