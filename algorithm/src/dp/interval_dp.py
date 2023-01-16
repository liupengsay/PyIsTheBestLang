"""

"""
"""
算法：区间DP
功能：xxx
题目：

前缀和优化区间DP（需要在状态转移的时候更新代价距离）
L2478 完美分割的方案数（https://leetcode.cn/problems/number-of-beautiful-partitions/）
L2463 最小移动总距离（https://leetcode.cn/problems/minimum-total-distance-traveled/）

预处理区间DP（需要预处理一个DP再计算最终DP）
L2472 不重叠回文子字符串的最大数目（https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/）回文子串判定DP加线性DP
L2430 对字母串可执行的最大删除数（https://leetcode.cn/problems/maximum-deletions-on-a-string/）最长公共前缀DP加线性DP
P1521 求逆序对（https://www.luogu.com.cn/problem/P1521）使用归并排序计算移动次数，也可以使用倍增的树状数组
P1775 石子合并（弱化版）（https://www.luogu.com.cn/problem/P1775）典型区间DP和前缀和预处理
P2426 删数（https://www.luogu.com.cn/problem/P2426）典型区间DP
P2690 [USACO04NOV]Apple Catching G（https://www.luogu.com.cn/problem/P2690）区间DP记忆化搜索模拟
P1435 [IOI2000] 回文字串（https://www.luogu.com.cn/problem/P1435）典型区间DP求最长不连续回文子序列

P1388 算式（https://www.luogu.com.cn/problem/P1388）回溯枚举符号组合，再使用区间DP进行最大值求解
P1103 书本整理（https://www.luogu.com.cn/problem/P1103）三维DP
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

MOD = 10 ** 9 + 7

class Solution:
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        # L2463
        robot.sort()
        factory.sort()
        m, n = len(factory), len(robot)
        dp = [[float("inf")]*(n+1) for _ in range(m+1)]
        dp[0][0] = 0
        for i in range(m):
            for j in range(n+1):
                if dp[i][j] < float("inf"):
                    dp[i+1][j] = min(dp[i+1][j], dp[i][j])
                    cost = 0
                    for k in range(1, factory[i][1]+1):
                        if j+k-1<n:
                            cost += abs(factory[i][0]-robot[j+k-1])
                            dp[i+1][j+k] = min(dp[i+1][j+k], dp[i][j]+cost)
                        else:
                            break
        return dp[-1][-1]


class Solution:
    def beautifulPartitions(self, s: str, k: int, minLength: int) -> int:

        # L2478
        start = set("2357")
        if s[0] not in start:
            return 0
        n = len(s)
        dp = [[0] * n for _ in range(k)]
        for i in range(n):
            if i + 1 >= minLength and s[i] not in start:
                dp[0][i] = 1

        for j in range(1, k):
            pre = 0
            x = 0
            for i in range(n):
                while x <= i - minLength and s[x]:
                    if s[x] not in start and s[x + 1] in start:
                        pre += dp[j - 1][x]
                        pre %= MOD
                    x += 1
                if s[i] not in start:
                    dp[j][i] = pre
        return dp[-1][-1]


class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        # L2472
        n = len(s)
        res = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            res[i][i] = 1
            if i + 1 < n:
                res[i][i + 1] = 1 if s[i] == s[i + 1] else 0
            for j in range(i + 2, n):
                if s[i] == s[j] and res[i + 1][j - 1]:
                    res[i][j] = 1

        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in range(0, i - k + 2):
                if i - j + 1 >= k and res[j][i]:
                    dp[i + 1] = max(dp[i + 1], dp[j] + 1)
        return dp[-1]
