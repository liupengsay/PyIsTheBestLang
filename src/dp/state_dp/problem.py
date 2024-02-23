"""
Algorithm：state_dp|outline_dp、memory_search|refresh_table|fill_table
Description：state_dp|dfs|back_track|brute_force|sub_set|bit_operation|brute_force

====================================LeetCode====================================
465（https://leetcode.cn/problems/optimal-account-balancing/）brute_force|sub_set|state_dp
1349（https://leetcode.cn/problems/maximum-students-taking-exam/）brute_force|state_dp
1723（https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/）bit_operation|minimum_maximum|brute_force|classical|sub_set
1986（https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）sub_set|preprocess|brute_force|state_dp
698（https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/））sub_set|preprocess|brute_force|state_dp
2172（https://leetcode.cn/problems/maximum-and-sum-of-array/）bit_operation|state_dp|3-base|state_dp
1255（https://leetcode.cn/problems/maximum-score-words-formed-by-letters/）state_dp
2403（https://leetcode.cn/problems/minimum-time-to-kill-all-monsters/）state_dp
1681（https://leetcode.cn/problems/minimum-incompatibility/）state_dp|group_bag_dp|state_dp|comb
1125（https://leetcode.cn/problems/smallest-sufficient-team/）state_dp
1467（https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/）memory_search|counter
1531（https://leetcode.cn/problems/string-compression-ii/submissions/）liner_dp|implemention
1595（https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/）state_dp
1655（https://leetcode.cn/problems/distribute-repeating-integers/）state_dp
1879（https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/）state_dp
2019（https://leetcode.cn/problems/the-score-of-students-solving-math-expression/）memory_search|fill_table
943（https://leetcode.cn/problems/find-the-shortest-superstring/）string|greedy|state_dp
1434（https://leetcode.cn/problems/number-of-ways-to-wear-different-hats-to-each-other/description/）state_dp|reverse_thinking
847（https://leetcode.cn/problems/shortest-path-visiting-all-nodes/）shortest_path|floyd|dijkstra|preprocess|state_dp
2741（https://leetcode.cn/problems/special-permutations/description/）state_dp
2305（https://leetcode.cn/problems/fair-distribution-of-cookies/description/）classical|state_dp|brute_force|sub_set
980（https://leetcode.cn/problems/unique-paths-iii/description/）classical|state_dp|back_track
2571（https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/description/）brain_teaser|memory_search

=====================================LuoGu======================================
P1896（https://www.luogu.com.cn/problem/P1896）brute_force|state_dp
P2704（https://www.luogu.com.cn/problem/P2704）state_dp

P2196（https://www.luogu.com.cn/problem/P2196）longest_path|state_dp
P1690（https://www.luogu.com.cn/problem/P1690）shortest_path|state_dp
P1294（https://www.luogu.com.cn/problem/P1294）state_dp|longest_diameter
P1123（https://www.luogu.com.cn/problem/P1123）state_dp
P1433（https://www.luogu.com.cn/problem/P1433）state_dp
P1896（https://www.luogu.com.cn/problem/P1896）state_dp
P1556（https://www.luogu.com.cn/problem/P1556）state_dp|shortest_path|specific_plan
P3052（https://www.luogu.com.cn/problem/P3052）state_dp|matrix_dp
P5997（https://www.luogu.com.cn/problem/P5997）greedy|bag_dp|state_dp
P6883（https://www.luogu.com.cn/problem/P6883）classical|state_dp
P8687（https://www.luogu.com.cn/problem/P8687）state_dp|bag_dp
P8733（https://www.luogu.com.cn/problem/P8733）floyd|shortest_path|state_dp

===================================CodeForces===================================
580D（https://codeforces.com/problemset/problem/580/D）state_dp
165E（https://codeforces.com/problemset/problem/165/E）liner_dp|state_dp|brute_force
11D（https://codeforces.com/contest/11/problem/D）state_dp|undirected|counter
1102F（https://codeforces.com/contest/1102/problem/F）state_dp|classical|brute_force|fill_table|refresh_table

=====================================AtCoder====================================
ABC332E（https://atcoder.jp/contests/abc332/tasks/abc332_e）math|state_dp|classical
ABC338F（https://atcoder.jp/contests/abc338/tasks/abc338_f）floyd|shortest_path|state_dp|fill_table|refresh_table|classical

=====================================AcWing=====================================
3735（https://www.acwing.com/problem/content/3738/）reverse_order|state_dp|specific_plan


"""
import heapq
import math
from collections import Counter
from functools import lru_cache
from functools import reduce
from itertools import combinations, accumulate
from operator import or_
from typing import List

from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1681(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-incompatibility/
        tag: state_dp|group_bag_dp|state_dp|comb|fill_table
        """
        n = len(nums)
        if n % k:
            return -1
        group = dict()
        ceil = [0] * (1 << n)
        floor = [inf] * (1 << n)
        ind = {1 << i: i for i in range(n)}
        m = n // k
        for i in range(1, 1 << n):
            x = nums[ind[i & (-i)]]
            ceil[i] = max(ceil[i & (i - 1)], x)
            floor[i] = min(floor[i & (i - 1)], x)
            if i.bit_count() == m:
                lst = [nums[j] for j in range(n) if i & (1 << j)]
                if len(set(lst)) == m:
                    group[i] = ceil[i] - floor[i]

        dp = [inf] * (1 << n)
        dp[0] = 0
        for i in range(1 << n):
            if dp[i] == inf:
                continue
            not_seen = {nums[j]: j for j in range(n) if not i & (1 << j)}
            mask = sum(1 << x for x in not_seen.values())
            sub = mask
            while sub:
                if sub in group:
                    dp[i | sub] = min(dp[i | sub], dp[i] + group[sub])
                sub = (sub - 1) & mask
        return dp[-1] if dp[-1] < inf else -1

    @staticmethod
    def lc_1723(jobs: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/
        tag: bit_operation|minimum_maximum|brute_force|classical|sub_set|refresh_table
        """
        n = len(jobs)
        ind = {1 << i: i for i in range(n)}
        cost = [0] * (1 << n)
        for i in range(1, 1 << n):
            cost[i] = cost[i & (i - 1)] + jobs[ind[i & (-i)]]
        pre = cost[:]
        cur = cost[:]
        for _ in range(k - 1):
            for i in range(1, 1 << n):
                sub = i
                while sub:
                    if cost[sub] < cur[i] and pre[i ^ sub] < cur[i]:
                        cur[i] = max(pre[i ^ sub], cost[sub])
                    sub = (sub - 1) & i
            for i in range(1 << n):
                pre[i] = cur[i]
        return pre[-1]

    @staticmethod
    def lc_1879_1(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/
        tag: state_dp|refresh_table
        """
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
    def lc_1879_2(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/
        tag: state_dp|fill_table

        """
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
        """
        url: https://codeforces.com/problemset/problem/165/E
        tag: liner_dp|state_dp|brute_force
        """
        # 线性state_dpDP，类似子集思想求解可能存在的与为0的数对
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
        """
        url: https://codeforces.com/problemset/problem/580/D
        tag: state_dp
        """
        # bitmaskbit_operationstate_dp转移，从 1 少的状态向多的转移，并brute_force前一个 1 的位置增益
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
        """
        url: https://leetcode.cn/problems/shortest-path-visiting-all-nodes/
        tag: shortest_path|floyd|dijkstra|preprocess|state_dp
        """
        # shortest_pathFloyd或者Dijkstrapreprocessshortest_path|state_dp
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
        """
        url: https://leetcode.cn/problems/maximum-students-taking-exam/
        tag: brute_force|state_dp
        """
        # 考试就座state_dp DP

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
        """
        url: https://leetcode.cn/problems/number-of-ways-to-wear-different-hats-to-each-other/description/
        tag: state_dp|reverse_thinking
        """
        # state_compressreverse_thinking，memory_search实现
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
        """
        url: https://leetcode.cn/problems/number-of-ways-to-wear-different-hats-to-each-other/description/
        tag: state_dp|reverse_thinking
        """
        # state_compressreverse_thinking，fill_table迭代实现
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
        """
        url: https://leetcode.cn/problems/minimum-time-to-kill-all-monsters/
        tag: state_dp
        """
        # state_dpDP数组形式
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
        """
        url: https://leetcode.cn/problems/minimum-time-to-kill-all-monsters/
        tag: state_dp
        """

        # state_dpDPmemory_search形式

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
        """
        url: https://www.luogu.com.cn/problem/P1896
        tag: state_dp
        """
        # state_compress迭代写法
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
        """
        url: https://codeforces.com/contest/11/problem/D
        tag: state_dp|undirected|counter
        """
        # state_compress无向图简单环counter
        n, m = ac.read_list_ints()

        # build_graph|
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
        """
        url: https://www.luogu.com.cn/problem/P1433
        tag: state_dp
        """
        # state_dp
        n = ac.read_int()
        lst = [[0, 0]]
        for _ in range(n):
            x, y = [float(w) for w in ac.read_list_strs() if w]
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
        """
        url: https://www.luogu.com.cn/problem/P1556
        tag: state_dp|shortest_path|specific_plan
        """
        # state_dpshortest_path
        n = ac.read_int()
        # 增|虚拟的起终点
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)] + [[0, 0]]
        n += 2
        # 根据题意build_graph|，表示起终点与方向
        dct = [[] for _ in range(n)]
        for i in range(n):
            a, b = nums[i]
            for j in range(n):
                if i != j:
                    # 只有在同一行或者同一列时可以build_graph|连边
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
        """
        url: https://www.luogu.com.cn/problem/P3052
        tag: state_dp|matrix_dp
        """
        # state_dp DP 二维优化
        n, w = ac.read_list_ints()
        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())
        f = [inf] * (1 << n)  # 当前状态下的分组
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
        """
        url: https://www.luogu.com.cn/problem/P5997
        tag: greedy|bag_dp|state_dp
        """
        # greedy背包与state_compress 结合
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
        """
        url: https://www.luogu.com.cn/problem/P6883
        tag: classical|state_dp
        """
        # classicalstate_compress
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
        """
        url: https://www.luogu.com.cn/problem/P8687
        tag: state_dp|bag_dp
        """
        # state_dp 结合背包 DP 思想
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
        """
        url: https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/
        tag: memory_search|counter
        """

        # memory_search与组合mathcounter

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
        """
        url: https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/
        tag: state_dp
        """
        # state_dp，需要一点变形
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
        """
        url: https://leetcode.cn/problems/distribute-repeating-integers/
        tag: state_dp
        """

        # 线性索引|brute_force子集state_compress
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
        """
        url: https://leetcode.cn/problems/the-score-of-students-solving-math-expression/
        tag: memory_search|fill_table
        """

        # 类似divide_and_conquer的思想memory_search
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
        """
        url: https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/
        tag: sub_set|preprocess|brute_force|state_dp
        """
        # preprocess子集后memory_search状态转移，子集brute_force，也可两个状态
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
        """
        url: https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/
        tag: sub_set|preprocess|brute_force|state_dp
        """

        # preprocess子集后memory_search状态转移，子集brute_force，也可两个状态

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
    def abc_332e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc332/tasks/abc332_e
        tag: math|state_dp|classical
        """
        n, d = ac.read_list_ints()
        nums = ac.read_list_ints()
        cost = [0] * (1 << n)
        ind = {1 << i: i for i in range(n)}
        for i in range(1, 1 << n):
            cost[i] = cost[i & (i - 1)] + nums[ind[i & (-i)]]

        dp = [x * x for x in cost]
        for _ in range(d - 1):
            for state in range((1 << n) - 1, -1, -1):
                mask = state
                while mask:
                    c = dp[state ^ mask] + cost[mask] * cost[mask]
                    if c < dp[state]:
                        dp[state] = c
                    mask = (mask - 1) & state

        ans = dp[-1]
        s = sum(nums)
        ac.st(((d * ans) - s * s) / d / d)
        return

    @staticmethod
    def ac_3735(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3738/
        tag: reverse_order|state_dp|specific_plan
        """
        # reverse_order|state_compress与输出specific_plan
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
        """
        url: https://leetcode.cn/problems/maximum-and-sum-of-array/
        tag: bit_operation|state_dp|3-base|state_dp
        """

        # bit_operation和state_dp转移，三进制state_compress（天平就是三进制）

        def get_k_bin_of_n(n: int, k: int, m: int):  # 进制与数字转换state_compress
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

    @staticmethod
    def cf_1102f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1102/problem/F
        tag: state_dp|classical|brute_force|fill_table|refresh_table
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        if m == 1:
            ans = min(abs(grid[0][j + 1] - grid[0][j]) for j in range(n - 1))
            ac.st(ans)
            return

        cost = [[inf] * m for _ in range(m)]
        end = [[inf] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                cost[i][j] = cost[j][i] = min(abs(grid[i][x] - grid[j][x]) for x in range(n))
            if n > 1:
                for j in range(m):
                    if j != i:
                        end[i][j] = min(abs(grid[i][x - 1] - grid[j][x]) for x in range(1, n))
        ans = 0
        for i in range(m):
            dp = [[0] * (1 << m) for _ in range(m)]
            dp[i][1 << i] = inf
            for s in range(1, 1 << m):
                tmp_s = [y for y in range(m) if not s & (1 << y)]
                for x in range(m):
                    if dp[x][s]:
                        for y in tmp_s:
                            dp[y][s | (1 << y)] = ac.max(dp[y][s | (1 << y)], ac.min(dp[x][s], cost[x][y]))
            for x in range(m):
                cur = dp[x][-1]
                if n > 1:
                    cur = ac.min(cur, end[x][i])
                if cur > ans:
                    ans = cur
        ac.st(ans)
        return

    @staticmethod
    def abc_338f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc338/tasks/abc338_f
        tag: floyd|shortest_path|state_dp|fill_table|refresh_table|classical
        """
        n, m = ac.read_list_ints()
        dis = [[inf] * n for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            dis[u - 1][v - 1] = w
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dis[i][k] < inf and dis[k][j] < inf:
                        dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
        m = 1 << n
        dp = [inf] * m * n

        for i in range(n):
            dp[i * m + (1 << i)] = 0
        for s in range(1 << n):
            for j in range(n):
                if dp[j * m + s] == inf or not (s >> j) & 1:
                    continue
                for k in range(n):
                    if dis[j][k] == inf or (s >> k) & 1:
                        continue
                    dp[k * m + (s | (1 << k))] = min(dp[k * m + (s | (1 << k))], dp[j * m + s] + dis[j][k])
        ans = min(dp[i * m + m - 1] for i in range(n))
        ac.st(ans if ans < inf else "No")
        return
