



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

from algorithm.src.fast_io import FastIO

"""
算法：状态压缩DP
功能：使用二进制数字表示转移状态，计算相应的转移方程，通常可以先计算满足条件的子集，有时通过深搜回溯枚举全部子集的办法比位运算枚举效率更高
题目：

===================================力扣===================================
1349. 参加考试的最大学生数（https://leetcode.cn/problems/maximum-students-taking-exam/）按行状态枚举所有的摆放可能性
1723. 完成所有工作的最短时间（https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/）通过位运算枚举分配工作DP最小化的最大值
1986. 完成任务的最少工作时间段（https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）预处理计算子集后进行记忆化状态转移
698. 划分为k个相等的子集（https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/）预处理计算子集后进行记忆化状态转移
2172. 数组的最大与和（https://leetcode.cn/problems/maximum-and-sum-of-array/）使用位运算和状态压缩进行转移
1255. 得分最高的单词集合（https://leetcode.cn/problems/maximum-score-words-formed-by-letters/）状压DP
2403. 杀死所有怪物的最短时间（https://leetcode.cn/problems/minimum-time-to-kill-all-monsters/）状压DP
1681. 最小不兼容性（https://leetcode.cn/problems/minimum-incompatibility/）状态压缩分组DP，状态压缩和组合数选取结合使用

===================================洛谷===================================
P1896 互不侵犯（https://www.luogu.com.cn/problem/P1896）按行状态与行个数枚举所有的摆放可能性
P2704 炮兵阵地（https://www.luogu.com.cn/problem/P2704）记录两个前序状态进行转移

P2196 [NOIP1996 提高组] 挖地雷（https://www.luogu.com.cn/problem/P2196）有向图最长路径加状压DP
P1690 贪婪的Copy（https://www.luogu.com.cn/problem/P1690）最短路加状压DP
P1294 高手去散步（https://www.luogu.com.cn/problem/P1294）图问题使用状压DP求解最长直径
P1123 取数游戏（https://www.luogu.com.cn/problem/P1123）类似占座位的经典状压DP

================================CodeForces================================
D. Kefa and Dishes（https://codeforces.com/problemset/problem/580/D）状态压缩DP结合前后相邻的增益计算最优解
E. Compatible Numbers（https://codeforces.com/problemset/problem/165/E）线性DP，状态压缩枚举，类似子集思想求解可能存在的与为0的数对

参考：OI WiKi（xx）
"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1681(self, nums: List[int], k: int) -> int:
        # 模板：状态压缩和组合数选取结合使用

        @lru_cache(None)
        def dfs(state):
            if not state:
                return 0

            dct = dict()
            for j in range(n):
                if state & (1 << j):
                    dct[nums[j]] = j
            if len(dct) < m:
                return inf
            res = inf
            for item in combinations(list(dct.keys()), m):
                cur = max(item) - min(item)
                nex = state
                for num in item:
                    nex ^= (1 << dct[num])
                x = dfs(nex) + cur
                res = res if res < x else x
            return res

        n = len(nums)
        if n % k:
            return -1
        inf = float("inf")
        m = n // k
        ans = dfs((1 << n) - 1)
        return ans if ans < inf else -1

    @staticmethod
    def cf_165e(ac=FastIO()):
        # 模板：线性状态压缩DP，类似子集思想求解可能存在的与为0的数对
        n = ac.read_int()
        nums = ac.read_list_ints()
        ceil = max(nums).bit_length()
        dp = [-1] * (1 << ceil)
        for num in nums:
            dp[num] = num

        for i in range(1, 1 << ceil):
            if dp[i] == -1:
                for j in range(i.bit_length()):
                    if i & (1 << j) and dp[i ^ (1 << j)] != -1:
                        dp[i] = dp[i ^ (1 << j)]
                        break

        ans = [-1] * n
        for i in range(n):
            num = nums[i]
            x = num ^ ((1 << ceil) - 1)
            ans[i] = dp[x]
        ac.lst(ans)
        return

    @staticmethod
    def cf_580d(ac):

        # 模板：bitmask位运算状态压缩转移，从 1 少的状态向多的转移，并枚举前一个 1 的位置计算增益
        n, m, k = ac.read_ints()
        ind = {1 << i: i for i in range(n + 1)}
        nums = ac.read_list_ints()
        dp = [[0] * (n + 1) for _ in range(1 << n)]
        edge = [[0] * (n + 1) for _ in range(n + 1)]
        for _ in range(k):
            x, y, c = ac.read_ints()
            x -= 1
            y -= 1
            edge[x][y] = c

        ans = 0
        for i in range(1, 1 << n):
            if bin(i).count("1") > m:
                continue
            res = 0
            mask = i
            while mask:
                j = ind[mask & (-mask)]
                cur = max(dp[i ^ (1 << j)][k] + edge[k][j] for k in range(n) if i & (1 << k)) + nums[j]
                res = ac.max(res, cur)
                mask &= (mask - 1)
                dp[i][j] = cur
            if bin(i).count("1") == m:
                ans = ac.max(ans, res)
        ac.st(ans)
        return

    @staticmethod
    def lc_1349(seats: List[List[str]]) -> int:

        # 模板：经典考试就座状态压缩 DP

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
    def lg_p1896(n, k):
        # 模板：经典国王摆放状态压缩 DP
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
                        if any(item[j + 1] - item[j] == 1 for j in range(length - 1)):
                            continue
                        nex = sum(1 << w for w in item)
                        res += dfs(nex, i + 1, x + length)
            return res

        return dfs(0, 0, 0)

    @staticmethod
    def lc_2403_1(power: List[int]) -> int:
        # 模板：状态压缩DP数组形式
        m = len(power)
        dp = [0] * (1 << m)
        for state in range(1, 1 << m):
            gain = m - state.bit_count() + 1
            res = float("inf")
            for i in range(m):
                if state & (1 << i):
                    cur = (power[i] + gain - 1) // gain + dp[state ^ (1 << i)]
                    res = res if res < cur else cur
            dp[state] = res
        return dp[-1]
    
    @staticmethod
    def lc_2403_2(power: List[int]) -> int:
        # 模板：状态压缩DP记忆化形式
        
        @lru_cache(None)
        def dfs(state):
            if not state:
                return 0
            gain = m - bin(state).count("1") + 1
            res = float("inf")
            for i in range(m):
                if state & (1 << i):
                    cur = math.ceil(power[i] / gain) + dfs(state ^ (1 << i))
                    res = res if res < cur else cur
            return res

        m = len(power)
        return dfs((1 << m) - 1)
    
    

class TestGeneral(unittest.TestCase):

    def test_state_dp(self):
        sd = Solution()
        assert sd.lg_p1896(9, 12) == 50734210126

        seats = [["#", ".", ".", ".", "#"], [".", "#", ".", "#", "."], [".", ".", "#", ".", "."], [".", "#", ".", "#", "."], ["#", ".", ".", ".", "#"]]
        assert sd.main_l1349(seats) == 10

        return


if __name__ == '__main__':
    unittest.main()
