



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
from math import inf

from algorithm.src.fast_io import FastIO

"""
算法：状态压缩DP、轮廓线DP、记忆化搜索DP
功能：使用二进制数字表示转移状态，计算相应的转移方程，通常可以先计算满足条件的子集，有时通过深搜回溯枚举全部子集的办法比位运算枚举效率更高
题目：

===================================力扣===================================
465. 最优账单平衡（https://leetcode.cn/problems/optimal-account-balancing/）经典枚举子集状压DP
1349. 参加考试的最大学生数（https://leetcode.cn/problems/maximum-students-taking-exam/）按行状态枚举所有的摆放可能性
1723. 完成所有工作的最短时间（https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/）通过位运算枚举分配工作DP最小化的最大值
1986. 完成任务的最少工作时间段（https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）预处理计算子集后进行记忆化状态转移，经典子集枚举
698. 划分为k个相等的子集（https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/）预处理计算子集后进行记忆化状态转移
2172. 数组的最大与和（https://leetcode.cn/problems/maximum-and-sum-of-array/）使用位运算和状态压缩进行转移
1255. 得分最高的单词集合（https://leetcode.cn/problems/maximum-score-words-formed-by-letters/）状压DP
2403. 杀死所有怪物的最短时间（https://leetcode.cn/problems/minimum-time-to-kill-all-monsters/）状压DP
1681. 最小不兼容性（https://leetcode.cn/problems/minimum-incompatibility/）状态压缩分组DP，状态压缩和组合数选取结合使用
1125. 最小的必要团队（https://leetcode.cn/problems/smallest-sufficient-team/）经典状压DP
1467. 两个盒子中球的颜色数相同的概率（https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/）记忆化搜索
1531. 压缩字符串 II（https://leetcode.cn/problems/string-compression-ii/submissions/）线性DP模拟
1595. 连通两组点的最小成本（https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/）经典状压DP
1655. 分配重复整数（https://leetcode.cn/problems/distribute-repeating-integers/）经典状压 DP
1879. 两个数组最小的异或值之和（https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/）经典状压 DP

===================================洛谷===================================
P1896 互不侵犯（https://www.luogu.com.cn/problem/P1896）按行状态与行个数枚举所有的摆放可能性
P2704 炮兵阵地（https://www.luogu.com.cn/problem/P2704）记录两个前序状态进行转移

P2196 [NOIP1996 提高组] 挖地雷（https://www.luogu.com.cn/problem/P2196）有向图最长路径加状压DP
P1690 贪婪的Copy（https://www.luogu.com.cn/problem/P1690）最短路加状压DP
P1294 高手去散步（https://www.luogu.com.cn/problem/P1294）图问题使用状压DP求解最长直径
P1123 取数游戏（https://www.luogu.com.cn/problem/P1123）类似占座位的经典状压DP
P1433 吃奶酪（https://www.luogu.com.cn/problem/P1433）状压DP
P1896 [SCOI2005] 互不侵犯（https://www.luogu.com.cn/problem/P1896）状压DP
P1556 幸福的路（https://www.luogu.com.cn/problem/P1556）状态压缩计算最短路
P3052 [USACO12MAR]Cows in a Skyscraper G（https://www.luogu.com.cn/problem/P3052）经典状态压缩 DP 使用二维优化
P5997 [PA2014]Pakowanie（https://www.luogu.com.cn/problem/P5997）经典贪心背包与状压 DP 结合
P6883 [COCI2016-2017#3] Kroničan（https://www.luogu.com.cn/problem/P6883）典型状压 DP 
P8687 [蓝桥杯 2019 省 A] 糖果（https://www.luogu.com.cn/problem/P8687）经典状压 DP 结合背包 DP 思想
P8733 [蓝桥杯 2020 国 C] 补给（https://www.luogu.com.cn/problem/P8733）使用Floyd最短路计算并使用状压 DP

================================CodeForces================================
D. Kefa and Dishes（https://codeforces.com/problemset/problem/580/D）状态压缩DP结合前后相邻的增益计算最优解
E. Compatible Numbers（https://codeforces.com/problemset/problem/165/E）线性DP，状态压缩枚举，类似子集思想求解可能存在的与为0的数对
D. A Simple Task（https://codeforces.com/contest/11/problem/D）状压DP，无向图简单环计数



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
        inf = inf
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
    def lc_2403_1(power: List[int]) -> int:
        # 模板：状态压缩DP数组形式
        m = len(power)
        dp = [0] * (1 << m)
        for state in range(1, 1 << m):
            gain = m - state.bit_count() + 1
            res = inf
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
            res = inf
            for i in range(m):
                if state & (1 << i):
                    cur = math.ceil(power[i] / gain) + dfs(state ^ (1 << i))
                    res = res if res < cur else cur
            return res

        m = len(power)
        return dfs((1 << m) - 1)

    @staticmethod
    def lg_p1896(ac=FastIO()):
        # 模板：状压DP迭代写法
        n, k = ac.read_ints()
        dp = [[[0]*(k+1) for _ in range(1<<n)] for _ in range(n+1)]
        dp[0][0][0] = 1

        for i in range(n):  # 行
            for j in range(1 << n):
                for num in range(k+1):
                    cur = [x for x in range(n) if not j & (1<<x) and (x==0 or not j & (1<<(x-1))) and (x==n-1 or not j &(1<<(x+1)))]
                    for y in range(1, len(cur)+1):
                        if num + y <= k:
                            for item in combinations(cur, y):
                                if all(item[p]-item[p-1] != 1 for p in range(1, y)):
                                    state = reduce(or_, [1 << z for z in item])
                                    dp[i+1][state][num+y] += dp[i][j][num]
                    dp[i+1][0][num] += dp[i][j][num]
        ans = sum(dp[n][j][k] for j in range(1<<n))
        ac.st(ans)
        return

    @staticmethod
    def cf_11d(ac=FastIO()):

        # 模板：状压DP无向图简单环计数
        n, m = ac.read_ints()

        # 建图
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        # 初始化
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1

        ans = 0
        for i in range(1, 1 << n):  # 经过的点状态，lowest_bit为起点
            for j in range(n):
                if not dp[i][j]:
                    continue
                for k in dct[j]:
                    # 下一跳必须不能比起点序号小
                    if (i & -i) > (1 << k):
                        continue
                    if i & (1 << k):
                        # 访问过且是起点则形成环
                        if (i & -i) == 1 << k:
                            ans += dp[i][j]
                    else:
                        # 未访问过传到下一状态
                        dp[i ^ (1 << k)][k] += dp[i][j]

        # 去除一条边的环以及是无向图需要除以二
        ans = (ans - m) // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p1433(ac=FastIO()):
        # 模板：状压DP
        n = ac.read_int()
        lst = [[0, 0]]
        for _ in range(n):
            x, y = [float(w)
                    for w in sys.stdin.readline().strip().split() if w]
            if not x == y == 0:
                lst.append([x, y])

        n = len(lst)
        grid = [[0.0] * n for _ in range(n)]
        for i in range(n):
            a, b = lst[i]
            for j in range(i + 1, n):
                c, d = lst[j]
                cur = math.sqrt((a - c) * (a - c) + (b - d) * (b - d))
                grid[i][j] = cur
                grid[j][i] = cur

        dp = [[inf] * n for _ in range(1 << n)]
        for i in range((1 << n) - 1):
            for pre in range(n):
                if not i:
                    dp[i][pre] = 0
                    continue
                res = inf
                for j in range(n):
                    if i & (1 << j):
                        cur = dp[i ^ (1 << j)][j] + grid[pre][j]
                        res = ac.min(res, cur)
                dp[i][pre] = res
        ans = dp[(1 << n) - 2][0]
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1556(ac=FastIO()):
        # 模板：状态压缩计算最短路
        n = ac.read_int()
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)] + [[0, 0]]
        n += 2
        # 根据题意进行建图，表示起终点与方向
        dct = [[] for _ in range(n)]
        for i in range(n):
            a, b = nums[i]
            for j in range(n):
                if i != j:
                    c, d = nums[j]
                    if a == c:
                        dct[i].append([j, 4] if b < d else [j, 2])
                    if b == d:
                        dct[i].append([j, 1] if a < c else [j, 3])

        # 状态 当前点 方向
        dp = [[[0] * 5 for _ in range(n)] for _ in range((1 << n) - 1)]
        dp[0][n - 1] = [0, 1, 1, 1, 1]
        for state in range(1, (1 << n) - 1):
            for x in range(n):
                for f in range(5):
                    if x == n - 1:
                        dp[state][x][f] = 1 if not state else 0
                    res = 0
                    # 枚举上一个点与方向是否可以转移过来
                    for y, ff in dct[x]:
                        if state & (1 << y) and ff != f:
                            res += dp[state ^ (1 << y)][y][ff]
                    dp[state][x][f] = res
        ac.st(dp[(1 << n) - 1 - 1][0][0])
        return

    @staticmethod
    def lg_p3052(ac=FastIO()):
        # 模板：经典状态压缩 DP 使用二维优化
        n, w = ac.read_ints()
        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())
        f = [math.inf] * (1 << n)  # 当前状态下的分组
        f[0] = 1
        g = [0] * (1 << n)  # 当前状态下最后一组占用的重量
        for i in range(1, 1 << n):
            for j in range(n):
                if i & (1 << j):
                    pre = i ^ (1 << j)
                    # 装在当前最后一组
                    if g[pre] + nums[j] <= w and [f[i], g[i]] > [f[pre], g[pre] + nums[j]]:
                        f[i] = f[pre]
                        g[i] = g[pre] + nums[j]
                    # 新开组
                    elif [f[i], g[i]] > [f[pre] + 1, nums[j]]:
                        f[i] = f[pre] + 1
                        g[i] = nums[j]
        ac.st(f[-1])
        return

    @staticmethod
    def lg_p5997(ac=FastIO()):
        # 模板：经典贪心背包与状压 DP 结合
        n, m = ac.read_ints()
        a = ac.read_list_ints()
        c = ac.read_list_ints()
        c.sort(reverse=True)
        # 状态最少需要的背包数
        dp = [m + 1] * (1 << n)
        dp[0] = 0
        # 状态最新背包剩余的空间
        rest = [0] * (1 << n)
        for i in range(1, 1 << n):
            for j in range(n):
                if i & (1 << j):
                    dd, rr = dp[i ^ (1 << j)], rest[i ^ (1 << j)]
                    # 直接原装
                    if rr >= a[j]:
                        if dp[i] > dd or (dp[i] == dd and rest[i] < rr - a[j]):
                            dp[i] = dd
                            rest[i] = rr - a[j]
                    # 新增背包
                    if dd + 1 <= m:
                        rr = c[dd]
                        if rr >= a[j]:
                            if dp[i] > dd + 1 or (dp[i] == dd + 1 and rest[i] < rr - a[j]):
                                dp[i] = dd + 1
                                rest[i] = rr - a[j]
        ac.st(dp[-1] if dp[-1] < m + 1 else "NIE")
        return

    @staticmethod
    def lg_p6883(ac=FastIO()):
        # 模板：典型状压 DP
        n, k = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(n)]
        dp = [inf] * (1 << n)
        dp[-1] = 0
        ans = inf
        for i in range((1 << n) - 1, -1, -1):
            lst = [j for j in range(n) if (1 << j) & i]
            if len(lst) <= k:
                ans = ac.min(ans, dp[i])
                continue
            for j in lst:
                c = min(grid[j][k] for k in lst if k != j)
                dp[i ^ (1 << j)] = ac.min(dp[i ^ (1 << j)], dp[i] + c)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8687(ac=FastIO()):
        # 模板：经典状压 DP 结合背包 DP 思想
        n, m, k = ac.read_ints()
        dp = [inf] * (1 << m)
        dp[0] = 0
        for i in range(n):
            nums = ac.read_list_ints_minus_one()
            cur = reduce(or_, [1 << x for x in nums])
            for j in range(1 << m):
                if dp[j | cur] > dp[j] + 1:
                    dp[j | cur] = dp[j] + 1
        ac.st(dp[-1] if dp[-1] < inf else -1)
        return

    @staticmethod
    def lc_1655(nums: List[int], quantity: List[int]) -> bool:
        # 模板：经典线性索引加枚举子集状压DP
        @lru_cache(None)
        def dfs(i, state):
            if not state:
                return True
            if i == m:
                return False
            x = cnt[i]
            sub = state
            while sub:
                cost = sum(quantity[j] for j in range(n) if sub & (1 << j))
                if cost <= x and dfs(i + 1, state ^ sub):
                    return True
                sub = (sub - 1) & state
            return False

        cnt = list(Counter(nums).values())
        n = len(quantity)
        cnt = heapq.nlargest(n, cnt)
        m = len(cnt)
        return dfs(0, (1 << n) - 1)

    @staticmethod
    def lc_1986(tasks: List[int], sessionTime: int):
        # 模板：预处理计算子集后进行记忆化状态转移，经典子集枚举
        n = len(tasks)
        valid = [False] * (1 << n)
        for mask in range(1, 1 << n):
            needTime = 0
            for i in range(n):
                if mask & (1 << i):
                    needTime += tasks[i]
            if needTime <= sessionTime:
                valid[mask] = True

        f = [inf] * (1 << n)
        f[0] = 0
        for mask in range(1, 1 << n):
            subset = mask
            while subset:
                if valid[subset]:
                    a, b = f[mask], f[mask ^ subset] + 1
                    f[mask] = a if a < b else b
                subset = (subset - 1) & mask
        return f[(1 << n) - 1]


class TestGeneral(unittest.TestCase):

    def test_state_dp(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
