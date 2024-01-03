"""
Algorithm：bfs|deque_bfs|01-bfs|discretization_bfs|bound_bfs|coloring_method|odd_circle
Description：multi_source_bfs|bilateral_bfs|spfa|a-star|heuristic_search

====================================LeetCode====================================
1617（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）brute_force|tree_diameter


=====================================LuoGu======================================
P1099（https://www.luogu.com.cn/problem/P1099）tree_diameter|bfs|two_pointers|monotonic_queue|classical
P2491（https://www.luogu.com.cn/problem/P2491）tree_diameter|bfs|two_pointers|monotonic_queue|classical
P3304（https://www.luogu.com.cn/problem/P3304）tree_diameter

===================================CodeForces===================================
1805D（https://codeforces.com/problemset/problem/1805/D）tree_diameter

====================================AtCoder=====================================

=====================================AcWing=====================================

=====================================LibraryChcker=====================================
1（https://judge.yosupo.jp/problem/tree_diameter）tree_diameter

"""

from typing import List

from src.graph.tree_diameter.template import TreeDiameter
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO, inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1805d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1805/D
        tag: tree_diameter|classical|brain_teaser
        """
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append([v, 1])
            edge[v].append([u, 1])
        tree = TreeDiameter(edge)
        u, v = tree.get_diameter_info()[:2]
        dis1, _ = tree.get_bfs_dis(u)
        dis2, _ = tree.get_bfs_dis(v)
        diff = [0] * (n + 1)
        for i in range(n):
            diff[ac.max(dis1[i], dis2[i]) + 1] += 1
        diff[0] = 1
        for i in range(1, n + 1):
            diff[i] += diff[i - 1]
        ac.lst([ac.min(x, n) for x in diff[1:]])
        return

    @staticmethod
    def lg_p3304(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3304
        tag: tree_diameter|classical|hard|brain_teaser|hard|necessary_diameter_edge
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append((j, k))
            dct[j].append((i, k))

        x = 0
        dis = [inf] * n
        stack = [x]
        dis[x] = 0
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                if j != parent[i]:
                    parent[j] = i
                    dis[j] = dis[i] + w
                    stack.append(j)
        x = dis.index(max(dis))

        dis = [inf] * n
        stack = [x]
        dis[x] = 0
        parent = [-1] * n
        weight = [0] * n
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                if j != parent[i]:
                    parent[j] = i
                    weight[j] = w
                    dis[j] = dis[i] + w
                    stack.append(j)
        y = dis.index(max(dis))
        path = [y]
        path_weight = []
        while path[-1] != x:
            path_weight.append(weight[path[-1]])
            path.append(parent[path[-1]])
        for i in path:
            dis[i] = -1
        del parent, weight
        tot = sum(path_weight)
        m = len(path)
        right = 0
        pre = 0
        for i in range(1, m):
            pre += path_weight[i - 1]
            stack = [(path[i], 0, -1)]
            cur = 0
            while stack:
                x, d, fa = stack.pop()
                cur = ac.max(cur, d)
                for y, w in dct[x]:
                    if y != fa and dis[y] != -1:
                        stack.append((y, d + w, x))
            right = i
            if cur == tot - pre:
                break

        left = m - 1
        pre = 0
        for i in range(m - 2, -1, -1):
            pre += path_weight[i]
            stack = [(path[i], 0, -1)]
            cur = 0
            while stack:
                x, d, fa = stack.pop()
                cur = ac.max(cur, d)
                for y, w in dct[x]:
                    if y != fa and dis[y] != -1:
                        stack.append((y, d + w, x))
            left = i
            if cur == tot - pre:
                break

        ans = ac.max(0, right - left)
        ac.st(tot)
        ac.st(ans)
        return

    @staticmethod
    def lc_1617(n: int, edges: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/
        tag: brute_force|tree_diameter
        """
        ans = [0] * n
        for state in range(1, 1 << n):
            node = [i for i in range(n) if state & (1 << i)]
            ind = {num: i for i, num in enumerate(node)}
            m = len(node)
            dct = [[] for _ in range(m)]
            uf = UnionFind(m)
            for u, v in edges:
                u -= 1
                v -= 1
                if u in ind and v in ind:
                    dct[ind[u]].append([ind[v], 1])
                    dct[ind[v]].append([ind[u], 1])
                    uf.union(ind[u], ind[v])
            if uf.part != 1:
                continue
            tree = TreeDiameter(dct)
            ans[tree.get_diameter_info()[-1]] += 1
        return ans[1:]

    @staticmethod
    def library_checker_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/tree_diameter
        tag: tree_diameter
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y, c = ac.read_list_ints()
            dct[x].append((y, c))
            dct[y].append((x, c))
        tree = TreeDiameter(dct)
        _, _, path, d = tree.get_diameter_info()
        ac.lst([d, len(path)])
        ac.lst(path)
        return
