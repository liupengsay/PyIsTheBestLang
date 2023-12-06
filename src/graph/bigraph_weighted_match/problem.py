"""
Algorithm：bipartite_graph最大最小权值匹配、KM算法
Function：

====================================LeetCode====================================
1820（https://leetcode.com/problems/maximum-number-of-accepted-invitations/）匈牙利算法或者bipartite_graph最大权KM算法解决
1066（https://leetcode.com/problems/campus-bikes-ii/）bipartite_graph最小权KM算法解决
1947（https://leetcode.com/problems/maximum-compatibility-score-sum/）bipartite_graph最大权匹配，也可用状压DP

=====================================LuoGu======================================
3386（https://www.luogu.com.cn/problem/P3386）bipartite_graph最大匹配
6577（https://www.luogu.com.cn/problem/P6577）bipartite_graph最大权完美匹配
1894（https://www.luogu.com.cn/problem/P1894）bipartite_graph最大匹配，转换为网络流求解
3605（https://www.luogu.com.cn/problem/B3605）匈牙利算法bipartite_graph不带权最大匹配

===================================CodeForces===================================
1437C（https://codeforces.com/problemset/problem/1437/C）bipartite_graph最小权匹配

=====================================AcWing=====================================
4298（https://www.acwing.com/problem/content/4301/）匈牙利算法bipartite_graph模板题

================================LibraryChecker================================
1 Matching on Bipartite Graph（https://judge.yosupo.jp/problem/bipartitematching）unweighted match

"""

# EK算法
from collections import defaultdict
from typing import List

import numpy as np

from src.graph.bigraph_weighted_match.template import BipartiteMatching
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        n, m, k = ac.read_list_ints()
        bm = BipartiteMatching(n, m)
        for _ in range(k):
            a, b = ac.read_list_ints()
            bm.add_edge(a, b)

        matching = bm.solve()
        ac.st(len(matching))
        for a, b in matching:
            ac.lst([a, b])
        return

    @staticmethod
    def lc_1820(grid):
        # 匈牙利算法模板建图最大匹配
        m, n = len(grid), len(grid[0])
        dct = defaultdict(list)
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    dct[i].append(j)

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        match = [-1] * n
        ans = 0
        for i in range(m):
            visit = [False] * n
            if hungarian(i):
                ans += 1
        return ans

    @staticmethod
    def lc_1820_2(grid: List[List[int]]) -> int:
        # EK网络最大流算法模板建图最大匹配
        n = len(grid)
        m = len(grid[0])
        s = n + m + 1
        t = n + m + 2
        ek = EK(n + m, n * m, s, t)
        for i in range(n):
            for j in range(m):
                if grid[i][j]:
                    ek.add_edge(i, n + j, 1)
        for i in range(n):
            ek.add_edge(s, i, 1)
        for i in range(m):
            ek.add_edge(n + i, t, 1)
        return ek.pipline()

    @staticmethod
    def lc_1820_3(grid):
        # KM算法模板建图最大匹配
        n = max(len(grid), len(grid[0]))
        lst = [[0] * n for _ in range(n)]
        ind = 0
        for i in range(n):
            for j in range(n):
                try:
                    lst[i][j] = grid[i][j]
                except IndexError as _:
                    ind += 1

        arr = np.array(lst)
        km = KM()
        max_ = km.compute(arr)

        ans = 0
        for i, j in max_:
            ans += lst[i][j]
        return ans

    @staticmethod
    def lg_p1894(ac=FastIO()):
        # bipartite_graph最大权匹配（不带权也可以匈牙利算法）
        n, m = ac.read_list_ints()
        s = n + m + 1
        t = n + m + 2
        # 集合个数n与集合个数m
        ek = EK(n + m, n * m, s, t)
        for i in range(n):
            lst = ac.read_list_ints()[1:]
            for j in lst:
                # 增|边
                ek.add_edge(i, n + j - 1, 1)
        # 增|超级源点与汇点
        for i in range(n):
            ek.add_edge(s, i, 1)
        for i in range(m):
            ek.add_edge(n + i, t, 1)
        ac.st(ek.pipline())
        return

    @staticmethod
    def lc_1947(students: List[List[int]], mentors: List[List[int]]) -> int:
        # bipartite_graph最大权匹配，也可用状压DP
        m, n = len(students), len(students[0])

        # 建立权值矩阵
        grid = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                grid[i][j] = sum([students[i][k] == mentors[j][k] for k in range(n)])

        # KM算法bipartite_graph最大权匹配
        km = KM()
        max_ = km.compute(np.array(grid))
        return sum([grid[i][j] for i, j in max_])

    @staticmethod
    def lg_3386(ac=FastIO()):
        # 匈牙利算法bipartite_graph不带权最大匹配
        n, m, e = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        for _ in range(e):
            i, j = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[j].append(i)
        ac.st(Hungarian().dfs_recursion(n, m, dct))
        # ac.st(Hungarian().bfs_iteration(n, m, dct))
        return

    @staticmethod
    def lc_1066(workers: List[List[int]], bikes: List[List[int]]) -> int:
        # bipartite_graph最小权匹配
        n = len(workers)
        m = len(bikes)
        grid = [[0] * m for _ in range(m)]
        for i in range(n):
            for j in range(m):
                grid[i][j] = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1])

        a = np.array(grid)
        km = KM()
        min_ = km.compute(a.copy(), min=True)
        ans = 0
        for i, j in min_:
            ans += grid[i][j]
        return ans

    @staticmethod
    def ac_4298(ac=FastIO()):
        # 匈牙利算法bipartite_graph模板题
        m = ac.read_int()
        a = ac.read_list_ints()
        n = ac.read_int()
        b = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if abs(a[i] - b[j]) <= 1:
                    dct[i].append(j)
        ans = Hungarian().dfs_recursion(n, m, dct)
        ac.st(ans)
        return