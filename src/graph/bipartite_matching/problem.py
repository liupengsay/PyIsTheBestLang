"""
Algorithm：bipartite_graph|maximum_weight_match|minimum_weight_match|km|ek|unweighted
Description：

====================================LeetCode====================================
4（https://leetcode.cn/problems/broken-board-dominoes/）outline_dp|classical|hungarian
1820（https://leetcode.cn/problems/maximum-number-of-accepted-invitations/）hungarian|bipartite_graph|maximum_weight_match|km

=====================================LuoGu======================================
P3386（https://www.luogu.com.cn/problem/P3386）bipartite_graph|maximum_weight_match|km
P6577（https://www.luogu.com.cn/problem/P6577）bipartite_graph|maximum_weight_match|km
P1894（https://www.luogu.com.cn/problem/P1894）bipartite_graph|maximum_weight_match|km
B3605（https://www.luogu.com.cn/problem/B3605）hungarian|bipartite_graph|maximum_weight_match|km

===================================CodeForces===================================
1437C（https://codeforces.com/problemset/problem/1437/C）bipartite_graph|minimum_weight_match|km
1228D（https://codeforces.com/problemset/problem/1228/D）complete_tripartite|random_seed|random_hash|classical

=====================================AcWing=====================================
4298（https://www.acwing.com/problem/content/4301/）hungarian|bipartite_graph

================================LibraryChecker================================
1（https://judge.yosupo.jp/problem/bipartitematching）maximum_weight_match|bipartite_matching

================================CodeChef================================
1（https://www.codechef.com/problems/PERMMODK?tab=solution）BipartiteMatching

"""
import random
from typing import List

from src.graph.bipartite_matching.template import BipartiteMatching
from src.graph.network_flow.template import DinicMaxflowMinCut
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/bipartitematching
        tag: maximum_weight_match|bipartite_matching
        """
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
        tag: hungarian|bipartite_graph|maximum_weight_match|bipartite_matching
        """
        m, n = len(grid), len(grid[0])
        bm = BipartiteMatching(m, n)
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    bm.add_edge(i, j)
        matching = bm.solve()
        return len(matching)

    @staticmethod
    def lg_p1894(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1894
        tag: bipartite_graph|maximum_weight_match|km
        """
        n, m = ac.read_list_ints()
        bm = BipartiteMatching(n, m)
        for i in range(n):
            for j in ac.read_list_ints_minus_one()[1:]:
                bm.add_edge(i, j)
        matching = bm.solve()
        ac.st(len(matching))
        return

    @staticmethod
    def lg_3386(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3386
        tag: bipartite_graph|maximum_weight_match|km
        """
        n, m, e = ac.read_list_ints()
        bm = BipartiteMatching(n, m)
        for i in range(e):
            i, j = ac.read_list_ints_minus_one()
            bm.add_edge(i, j)
        matching = bm.solve()
        ac.st(len(matching))
        return

    @staticmethod
    def ac_4298(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4301/
        tag: hungarian|bipartite_graph
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        m = ac.read_int()
        b = ac.read_list_ints()
        bm = BipartiteMatching(n, m)
        for i in range(n):
            for j in range(m):
                if abs(a[i] - b[j]) <= 1:
                    bm.add_edge(i, j)
        matching = bm.solve()
        ac.st(len(matching))
        return

    @staticmethod
    def lc_4(n: int, m: int, broken: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/broken-board-dominoes/
        tag: outline_dp|classical|hungarian
        """
        m, n = n, m
        grid = [[0] * n for _ in range(m)]
        for i, j in broken:
            grid[i][j] = 1
        bm = BipartiteMatching(m * n, m * n)
        for i in range(m):
            for j in range(n):
                if not grid[i][j]:
                    for x, y in [[i + 1, j], [i, j + 1]]:
                        if 0 <= x < m and 0 <= y < n and not grid[x][y]:
                            bm.add_edge(i * n + j, x * n + y)
                            bm.add_edge(x * n + y, i * n + j)
        matching = bm.solve()
        return len(matching) // 2

    @staticmethod
    def abc_274g_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc274/tasks/abc274_g
        tag: bipartite_matching|minimum_point_cover|maximum_match|classical
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        row = [[-1] * n for _ in range(m)]
        r = flag = 0
        for i in range(m):
            if flag:
                r += 1
            flag = 0
            for j in range(n):
                if grid[i][j] == ".":
                    row[i][j] = r
                    flag = 1
                else:
                    if flag:
                        r += 1
                        flag = 0
        if flag:
            r += 1

        edge = set()
        c = flag = 0
        for j in range(n):
            if flag:
                c += 1
            flag = 0
            for i in range(m):
                if grid[i][j] == ".":
                    edge.add((row[i][j], c))
                    flag = 1
                else:
                    if flag:
                        c += 1
                        flag = 0
        if flag:
            c += 1
        bm = BipartiteMatching(r, c)
        for i, j in edge:
            bm.add_edge(i, j)
        matching = bm.solve()
        ac.st(len(matching))
        return

    @staticmethod
    def abc_274g_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc274/tasks/abc274_g
        tag: bipartite_matching|minimum_point_cover|maximum_match|classical
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        row = [[-1] * n for _ in range(m)]
        r = flag = 0
        for i in range(m):
            if flag:
                r += 1
            flag = 0
            for j in range(n):
                if grid[i][j] == ".":
                    row[i][j] = r
                    flag = 1
                else:
                    if flag:
                        r += 1
                        flag = 0
        if flag:
            r += 1

        edge = set()
        c = flag = 0
        for j in range(n):
            if flag:
                c += 1
            flag = 0
            for i in range(m):
                if grid[i][j] == ".":
                    edge.add((row[i][j], c))
                    flag = 1
                else:
                    if flag:
                        c += 1
                        flag = 0
        if flag:
            c += 1

        flow = DinicMaxflowMinCut(r + c + 2)
        for i in range(1, r + 1):
            flow.add_edge(r + c + 1, i, 1)
        for i in range(r + 1, r + c + 1):
            flow.add_edge(i, r + c + 2, 1)
        for i, j in edge:
            flow.add_edge(i + 1, j + r + 1, 1)
        ans = flow.max_flow_min_cut(r + c + 1, r + c + 2)
        ac.st(ans)
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1228/D
        tag: complete_tripartite|random_seed|random_hash|classical
        """
        n, m = ac.read_list_ints()
        nums = [random.getrandbits(32) for _ in range(n)]
        dct = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i] ^= nums[j]
            dct[j] ^= nums[i]
        lst = list(set(dct))
        if len(lst) != 3:
            ac.st(-1)
        elif 0 in lst:
            ac.st(-1)
        else:
            for i in range(n):
                dct[i] = lst.index(dct[i]) + 1
            ac.lst(dct)
        return

    @staticmethod
    def cf_1228d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1228/D
        tag: complete_tripartite|random_seed|random_hash|classical
        """
        n, m = ac.read_list_ints()
        nums = [random.getrandbits(32) for _ in range(n)]
        dct = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i] ^= nums[j]
            dct[j] ^= nums[i]
        lst = list(set(dct))
        if len(lst) != 3:
            ac.st(-1)
        elif 0 in lst:
            ac.st(-1)
        else:
            for i in range(n):
                dct[i] = lst.index(dct[i]) + 1
            ac.lst(dct)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/PERMMODK?tab=solution
        tag: BipartiteMatching
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()

            bm = BipartiteMatching(n, n)
            for i in range(n):
                for j in range(n):
                    if (i + 1) % k != (j + 1) % k:
                        bm.add_edge(i, j)
            matching = bm.solve()
            if len(matching) != n:
                ac.st(-1)
            else:
                ans = [x + 1 for _, x in matching]
                assert all(ans[i] % k != (i + 1) % k for i in range(n))
                ac.lst(ans)
        return
