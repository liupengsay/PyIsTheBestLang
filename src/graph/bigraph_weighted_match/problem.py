"""
Algorithm：bipartite_graph|maximum_weight_match|minimum_weight_match|km|ek
Description：

====================================LeetCode====================================
1820（https://leetcode.cn/problems/maximum-number-of-accepted-invitations/）hungarian|bipartite_graph|maximum_weight_match|km
1066（https://leetcode.cn/problems/campus-bikes-ii/）bipartite_graph|minimum_weight_match|km
1947（https://leetcode.cn/problems/maximum-compatibility-score-sum/）bipartite_graph|maximum_weight_match|state_compress

=====================================LuoGu======================================
P3386（https://www.luogu.com.cn/problem/P3386）bipartite_graph|maximum_weight_match|km
P6577（https://www.luogu.com.cn/problem/P6577）bipartite_graph|maximum_weight_match|km
P1894（https://www.luogu.com.cn/problem/P1894）bipartite_graph|maximum_weight_match|km
B3605（https://www.luogu.com.cn/problem/B3605）hungarian|bipartite_graph|maximum_weight_match|km

===================================CodeForces===================================
1437C（https://codeforces.com/problemset/problem/1437/C）bipartite_graph|minimum_weight_match|km

=====================================AcWing=====================================
4298（https://www.acwing.com/problem/content/4301/）hungarian|bipartite_graph

================================LibraryChecker================================
1 Matching on Bipartite Graph（https://judge.yosupo.jp/problem/bipartitematching）maximum_weight_match|km

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
        """
        url: https://leetcode.cn/problems/maximum-number-of-accepted-invitations/
        tag: hungarian|bipartite_graph|maximum_weight_match|km
        """
        # hungarian模板build_graph||maximum_weight_match|km
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
        """
        url: https://leetcode.cn/problems/maximum-number-of-accepted-invitations/
        tag: hungarian|bipartite_graph|maximum_weight_match|km
        """
        # EK网络最大流算法模板build_graph||maximum_weight_match|km
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
        """
        url: https://leetcode.cn/problems/maximum-number-of-accepted-invitations/
        tag: hungarian|bipartite_graph|maximum_weight_match|km
        """
        # KM算法模板build_graph||maximum_weight_match|km
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
        """
        url: https://www.luogu.com.cn/problem/P1894
        tag: bipartite_graph|maximum_weight_match|km
        """
        # bipartite_graphmaximum_weight_match（不带权也可以hungarian）
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
        """
        url: https://leetcode.cn/problems/maximum-compatibility-score-sum/
        tag: bipartite_graph|maximum_weight_match|state_compress
        """
        # bipartite_graphmaximum_weight_match，也可用state_compress
        m, n = len(students), len(students[0])

        # 建立权值矩阵
        grid = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                grid[i][j] = sum([students[i][k] == mentors[j][k] for k in range(n)])

        # KM算法bipartite_graphmaximum_weight_match
        km = KM()
        max_ = km.compute(np.array(grid))
        return sum([grid[i][j] for i, j in max_])

    @staticmethod
    def lg_3386(ac=FastIO()):
        # hungarianbipartite_graph不带权|maximum_weight_match|km
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
        """
        url: https://leetcode.cn/problems/campus-bikes-ii/
        tag: bipartite_graph|minimum_weight_match|km
        """
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
        """
        url: https://www.acwing.com/problem/content/4301/
        tag: hungarian|bipartite_graph
        """
        # hungarianbipartite_graph模板题
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
