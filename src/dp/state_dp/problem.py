"""
Algorithm：state_compressionDP、轮廓线DP、记忆化搜索DP、刷表法、填表法
Function：二进制数字表示转移状态，相应的转移方程，通常可以先满足条件的子集，有时通过深搜back_trackbrute_force全部子集的办法比bit_operationbrute_force效率更高

====================================LeetCode====================================
465（https://leetcode.com/problems/optimal-account-balancing/）brute_force子集状压DP
1349（https://leetcode.com/problems/maximum-students-taking-exam/）按行状态brute_force所有的摆放可能性
1723（https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/）通过bit_operationbrute_force分配工作DP最小化的最大值，brute_force子集预处理，brute_force子集模板
1986（https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）预处理子集后记忆化状态转移，子集brute_force，也可两个状态
698（https://leetcode.com/problems/partition-to-k-equal-sum-subsets/）预处理子集后记忆化状态转移
2172（https://leetcode.com/problems/maximum-and-sum-of-array/）bit_operation和state_compression转移，三进制状压DP（天平就是三进制）
1255（https://leetcode.com/problems/maximum-score-words-formed-by-letters/）状压DP
2403（https://leetcode.com/problems/minimum-time-to-kill-all-monsters/）状压DP
1681（https://leetcode.com/problems/minimum-incompatibility/）state_compression分组DP，state_compression和组合数选取结合
1125（https://leetcode.com/problems/smallest-sufficient-team/）状压DP
1467（https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/）记忆化搜索与组合mathcounter
1531（https://leetcode.com/problems/string-compression-ii/submissions/）线性DPimplemention
1595（https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points/）状压DP，需要一点变形
1655（https://leetcode.com/problems/distribute-repeating-integers/）状压 DP
1879（https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/）状压 DP
2019（https://leetcode.com/problems/the-score-of-students-solving-math-expression/）记忆化DP，可以刷表法与填表法迭代实现
943（https://leetcode.com/problems/find-the-shortest-superstring/）字符串greedy最短长度拼接状压DP
1434（https://leetcode.com/problems/number-of-ways-to-wear-different-hats-to-each-other/description/）状压DPreverse_thinking
847（https://leetcode.com/problems/shortest-path-visiting-all-nodes/）最短路Floyd或者Dijkstra预处理最短路|状压DP
2741（https://leetcode.com/problems/special-permutations/description/）状压DP
2305（https://leetcode.com/problems/fair-distribution-of-cookies/description/）典型状压DPbrute_force子集
980（https://leetcode.com/problems/unique-paths-iii/description/）典型状压DP或者back_track
2571（https://leetcode.com/problems/minimum-operations-to-reduce-an-integer-to-0/description/）思维题记忆化DP

=====================================LuoGu======================================
1896（https://www.luogu.com.cn/problem/P1896）按行状态与行个数brute_force所有的摆放可能性
2704（https://www.luogu.com.cn/problem/P2704）记录两个前序状态转移

2196（https://www.luogu.com.cn/problem/P2196）有向图最长路径|状压DP
1690（https://www.luogu.com.cn/problem/P1690）最短路|状压DP
1294（https://www.luogu.com.cn/problem/P1294）图问题状压DP求解最长直径
1123（https://www.luogu.com.cn/problem/P1123）类似占座位的状压DP
1433（https://www.luogu.com.cn/problem/P1433）状压DP
1896（https://www.luogu.com.cn/problem/P1896）状压DP
1556（https://www.luogu.com.cn/problem/P1556）state_compressionDP最短路方案数
3052（https://www.luogu.com.cn/problem/P3052）state_compression DP 二维优化
5997（https://www.luogu.com.cn/problem/P5997）greedy背包与状压 DP 结合
6883（https://www.luogu.com.cn/problem/P6883）典型状压 DP 
8687（https://www.luogu.com.cn/problem/P8687）状压 DP 结合背包 DP 思想
8733（https://www.luogu.com.cn/problem/P8733）Floyd最短路并状压 DP

===================================CodeForces===================================
580D（https://codeforces.com/problemset/problem/580/D）state_compressionDP结合前后相邻的增益最优解
165E（https://codeforces.com/problemset/problem/165/E）线性DP，state_compressionbrute_force，类似子集思想求解可能存在的与为0的数对
11D（https://codeforces.com/contest/11/problem/D）状压DP，无向图简单环counter
1294F（https://codeforces.com/contest/1294/problem/F）典型树的直径应用题

=====================================AcWing=====================================
3735（https://www.acwing.com/problem/content/3738/）倒序状压DP与输出specific_plan

"""
import heapq
import math
import sys
from collections import Counter
from functools import lru_cache
from functools import reduce
from itertools import combinations, accumulate
from math import inf
from operator import or_
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1681(nums: List[int], k: int) -> int:
        # state_compression和组合数选取结合

        @lru_cache(None)
        def dfs(state):
            if not state:
                return 0

            lst = []
            dct = dict()
            for j in range(n):
                if state & (1 << j) and nums[j] not in dct:
                    dct[nums[j]] = j
                    lst.append(nums[j])
            if len(dct) < m:
                return inf
            res = inf
            for item in combinations(lst, m):
                cur = max(item) - min(item)
                if cur > res:
                    break
                nex = state
                for num in item:
                    nex ^= (1 << dct[num])
                x = dfs(nex) + cur
                res = res if res < x else x
            return res

        n = len(nums)
        nums.sort()
        if n % k:
            return -1
        m = n // k
        ans = dfs((1 << n) - 1)
        return ans if ans < inf else -1

    @staticmethod
    def lc_1723(jobs: List[int], k: int) -> int:
        # 通过bit_operationbrute_force分配工作DP最小化的最大值，brute_force子集预处理

        @lru_cache(None)
        def dfs(i, state):
            if i == k:
                if not state:
                    return 0
                return inf
            res = inf
            sub = state
            while sub:
                cost = state_cost[sub]
                if cost < res:
                    nex = dfs(i + 1, state ^ sub)
                    if nex > cost:
                        cost = nex
                    if cost < res:
                        res = cost
                # brute_force子集模板
                sub = (sub - 1) & state
            return res

        n = len(jobs)
        state_cost = [0] * (1 << n)
        for x in range(1, 1 << n):
            state_cost[x] = sum(jobs[j] for j in range(n) if x & (1 << j))

        return dfs(0, (1 << n) - 1)

    @staticmethod
    def lc_1879_1(nums1: List[int], nums2: List[int]) -> int:

        # 记忆化深搜状压DP写法

        @lru_cache(None)
        def dfs(i, state):
            if i == n:
                return 0
            res = inf
            for j in range(n):
                if state & (1 << j):
                    cur = (nums1[i] ^ nums2[j]) + dfs(i + 1, state ^ (1 << j))
                    if cur < res:
                        res = cur
            return res

        n = len(nums1)
        return dfs(0, (1 << n) - 1)

    @staticmethod
    def lc_1879_2(nums1: List[int], nums2: List[int]) -> int:
        # 状压DP迭代写法，刷表法
        n = len(nums1)
        s = sum(nums1) + sum(nums2)
        dp = [s] * (1 << n)
        dp[0] = 0
        for state in range(1 << n):
            i = state.bit_count()
            for j in range(n):
                if not state & (1 << j):
                    a, b = dp[state | (1 << j)], (nums1[i] ^ nums2[j]) + dp[state]
                    dp[state | (1 << j)] = a if a < b else b
        return dp[-1]

    @staticmethod
    def lc_1879_3(nums1: List[int], nums2: List[int]) -> int:
        # 状压DP迭代写法，填表法
        n = len(nums1)
        s = sum(nums1) + sum(nums2)
        dp = [s] * (1 << n)
        dp[0] = 0
        for state in range(1, 1 << n):
            i = state.bit_count()
            for j in range(n):
                if state & (1 << j):
                    a, b = dp[state], (nums1[i - 1] ^ nums2[j]) + dp[state ^ (1 << j)]
                    dp[state] = a if a < b else b
        return dp[-1]

    @staticmethod
    def cf_165e(ac=FastIO()):
        # 线性state_compressionDP，类似子集思想求解可能存在的与为0的数对
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

        # bitmaskbit_operationstate_compression转移，从 1 少的状态向多的转移，并brute_force前一个 1 的位置增益
        n, m, k = ac.read_list_ints()
        ind = {1 << i: i for i in range(n + 1)}
        nums = ac.read_list_ints()
        dp = [[0] * (n + 1) for _ in range(1 << n)]
        edge = [[0] * (n + 1) for _ in range(n + 1)]
        for _ in range(k):
            x, y, c = ac.read_list_ints()
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
    def lc_847(graph: List[List[int]]) -> int:
        # 最短路Floyd或者Dijkstra预处理最短路|状压DP
        n = len(graph)
        dis = [[inf] * n for _ in range(n)]
        for i in range(n):
            for j in graph[i]:
                dis[i][j] = dis[j][i] = 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

        dp = [[inf] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 0

        for i in range(1 << n):
            for j in range(n):
                if dp[i][j] < inf and i & (1 << j):
                    for k in range(n):
                        if not i & (1 << k):
                            dp[i ^ (1 << k)][k] = min(dp[i ^ (1 << k)][k], dp[i][j] + dis[j][k])
        return min(dp[-1])

    @staticmethod
    def lc_1349(seats: List[List[str]]) -> int:

        # 考试就座state_compression DP

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
    def lc_1434_1(hats: List[List[int]]) -> int:
        # 状压DPreverse_thinking，记忆化实现
        mod = 10 ** 9 + 7
        n = len(hats)
        people = [[] for _ in range(40)]
        for u in range(n):
            for v in hats[u]:
                people[v - 1].append(u)

        @lru_cache(None)
        def dfs(state, i):
            if not state:
                return 1
            if i == 40:
                return 0
            res = dfs(state, i + 1)
            for j in people[i]:
                if (1 << j) & state:
                    res += dfs(state ^ (1 << j), i + 1)
                    res %= mod
            return res

        return dfs((1 << n) - 1, 0)

    @staticmethod
    def lc_1434_2(hats: List[List[int]]) -> int:
        # 状压DPreverse_thinking，填表法迭代实现
        mod = 10 ** 9 + 7
        n = len(hats)
        people = [[] for _ in range(40)]
        for u in range(n):
            for v in hats[u]:
                people[v - 1].append(u)

        dp = [[0] * (1 << n) for _ in range(41)]
        dp[0][0] = 1
        for i in range(40):
            for j in range(1 << n):
                dp[i + 1][j] = dp[i][j]
                for x in people[i]:
                    if (1 << x) & j:
                        dp[i + 1][j] += dp[i][j ^ (1 << x)]
                        dp[i + 1][j] %= mod

        return dp[-1][-1]

    @staticmethod
    def lc_2403_1(power: List[int]) -> int:
        # state_compressionDP数组形式
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
        # state_compressionDP记忆化形式

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
        # 状压DP迭代写法
        n, k = ac.read_list_ints()
        dp = [[[0] * (k + 1) for _ in range(1 << n)] for _ in range(n + 1)]
        dp[0][0][0] = 1

        for i in range(n):  # 行
            for j in range(1 << n):
                for num in range(k + 1):
                    cur = [x for x in range(n) if not j & (1 << x) and (x == 0 or not j & (1 << (x - 1))) and (
                            x == n - 1 or not j & (1 << (x + 1)))]
                    for y in range(1, len(cur) + 1):
                        if num + y <= k:
                            for item in combinations(cur, y):
                                if all(item[p] - item[p - 1] != 1 for p in range(1, y)):
                                    state = reduce(or_, [1 << z for z in item])
                                    dp[i + 1][state][num + y] += dp[i][j][num]
                    dp[i + 1][0][num] += dp[i][j][num]
        ans = sum(dp[n][j][k] for j in range(1 << n))
        ac.st(ans)
        return

    @staticmethod
    def cf_11d(ac=FastIO()):

        # 状压DP无向图简单环counter
        n, m = ac.read_list_ints()

        # 建图
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
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
        # 状压DP
        n = ac.read_int()
        lst = [[0, 0]]
        for _ in range(n):
            x, y = [float(w) for w in sys.stdin.readline().strip().split() if w]
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
        # state_compression最短路
        n = ac.read_int()
        # 增|虚拟的起终点
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)] + [[0, 0]]
        n += 2
        # 根据题意建图，表示起终点与方向
        dct = [[] for _ in range(n)]
        for i in range(n):
            a, b = nums[i]
            for j in range(n):
                if i != j:
                    # 只有在同一行或者同一列时可以建图连边
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
                    # brute_force上一个点与方向是否可以转移过来
                    for y, ff in dct[x]:
                        if state & (1 << y) and ff != f:
                            res += dp[state ^ (1 << y)][y][ff]
                    dp[state][x][f] = res
        #  0 表示初始任意不同于 1234 的方向总和
        ac.st(dp[(1 << n) - 1 - 1][0][0])
        return

    @staticmethod
    def lg_p3052(ac=FastIO()):
        # state_compression DP 二维优化
        n, w = ac.read_list_ints()
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
        # greedy背包与状压 DP 结合
        n, m = ac.read_list_ints()
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
        # 典型状压 DP
        n, k = ac.read_list_ints()
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
        # 状压 DP 结合背包 DP 思想
        n, m, k = ac.read_list_ints()
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
    def lc_1467(balls: List[int]) -> float:

        # 记忆化搜索与组合mathcounter

        @lru_cache(None)
        def dfs(i, s, c1, c2):
            if i == m:
                return c1 == c2 and s == n // 2
            res = 0
            t = pre[i] - s
            for x in range(0, balls[i] + 1):
                if s + x <= n // 2 and t + balls[i] - x <= n // 2:
                    cnt = math.comb((n // 2 - s), x) * math.comb((n // 2 - t), balls[i] - x)
                    res += dfs(i + 1, s + x, c1 + int(x > 0), c2 + int(x < balls[i])) * cnt
            return res

        m = len(balls)
        n = sum(balls)
        total = 1
        rest = n
        for num in balls:
            total *= math.comb(rest, num)
            rest -= num
        pre = list(accumulate(balls, initial=0))
        return dfs(0, 0, 0, 0) / total

    @staticmethod
    def lc_1595(cost: List[List[int]]) -> int:

        # 状压DP，需要一点变形
        m, n = len(cost), len(cost[0])
        low = [min(cost[i][j] for i in range(m)) for j in range(n)]

        @lru_cache(None)
        def dfs(i, state):
            if i == m:
                res = 0
                for j in range(n):
                    if not state & (1 << j):
                        res += low[j]
                return res
            return min(dfs(i + 1, state | (1 << j)) + cost[i][j] for j in range(n))

        return dfs(0, 0)

    @staticmethod
    def lc_1655(nums: List[int], quantity: List[int]) -> bool:
        # 线性索引|brute_force子集状压DP
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
    def lc_2019(s: str, answers: List[int]) -> int:
        # 类似divide_and_conquer的思想记忆化搜索
        @lru_cache(None)
        def dfs(state):
            if len(state) == 1:
                return set(state)
            m = len(state)
            cur = set()
            for i in range(m):
                if isinstance(state[i], str) and state[i] in "+*":
                    op = state[i]
                    pre = dfs(state[:i])
                    post = dfs(state[i + 1:])
                    for x in pre:
                        for y in post:
                            z = x + y if op == "+" else x * y
                            if z <= 1000:
                                cur.add(z)
            return cur

        lst = [int(w) if w.isnumeric() else w for w in s]
        res = dfs(tuple(lst))
        real = eval(s)
        return sum(5 if w == real else 2 if w in res else 0 for w in answers)

    @staticmethod
    def lc_1986_1(tasks: List[int], session: int):
        # 预处理子集后记忆化状态转移，子集brute_force，也可两个状态
        n = len(tasks)
        valid = [False] * (1 << n)
        for mask in range(1, 1 << n):
            cost = 0
            for i in range(n):
                if mask & (1 << i):
                    cost += tasks[i]
            if cost <= session:
                valid[mask] = True

        f = [inf] * (1 << n)
        f[0] = 0
        for mask in range(1, 1 << n):
            subset = mask
            while subset:  # 状压子集brute_force
                if valid[subset]:
                    a, b = f[mask], f[mask ^ subset] + 1
                    f[mask] = a if a < b else b
                subset = (subset - 1) & mask
        return f[(1 << n) - 1]

    @staticmethod
    def lc_1986_2(tasks: List[int], session: int):
        # 预处理子集后记忆化状态转移，子集brute_force，也可两个状态

        @lru_cache(None)
        def dfs(state, rest):
            if not state:
                return 0
            res = inf
            for i in range(n):
                if state & (1 << i):
                    if rest >= tasks[i]:
                        cur = dfs(state ^ (1 << i), rest - tasks[i])
                    else:
                        cur = 1 + dfs(state ^ (1 << i), session - tasks[i])
                    if cur < res:
                        res = cur
            return res

        n = len(tasks)
        return dfs((1 << n) - 1, 0)

    @staticmethod
    def ac_3735(ac=FastIO()):
        # 倒序状压DP与输出specific_plan
        n, m = ac.read_list_ints()
        if m == n * (n - 1) // 2:
            ac.st(0)
            return
        group = [0] * n
        for i in range(n):
            group[i] |= (1 << i)
        for _ in range(m):
            i, j = ac.read_list_ints()
            i -= 1
            j -= 1
            group[i] |= (1 << j)
            group[j] |= (1 << i)

        dp = [inf] * (1 << n)
        pre = [[-1, -1] for _ in range(1 << n)]
        for i in range(n):
            dp[group[i]] = 1
            pre[group[i]] = [i, -1]  # use, from

        for i in range(1 << n):
            if dp[i] == inf:
                continue

            for j in range(n):
                if i & (1 << j):
                    nex = i | group[j]
                    if dp[nex] > dp[i] + 1:
                        dp[nex] = dp[i] + 1
                        pre[nex] = [j, i]  # use, from

        s = (1 << n) - 1
        ans = []
        while s > 0:
            ans.append(pre[s][0] + 1)
            s = pre[s][1]
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def lc_2172(nums: List[int], num_slots: int) -> int:
        # bit_operation和state_compression转移，三进制状压DP（天平就是三进制）

        def get_k_bin_of_n(n: int, k: int, m: int):  # 进制与数字转换状压DP
            lst = []
            while n:
                lst.append(n % k)
                n //= k
            lst = lst + [0] * (m - len(lst))
            return lst

        length = len(nums)
        dp = [0] * (3 ** num_slots)
        for sub in range(3 ** num_slots):
            cnt = get_k_bin_of_n(sub, 3, num_slots)
            pre = sum(cnt)
            if pre >= length:
                continue
            for j in range(num_slots):
                if cnt[j] < 2:
                    cur = dp[sub] + (nums[pre] & (j + 1))
                    dp[sub + 3 ** j] = max(dp[sub + 3 ** j], cur)
        return max(dp)