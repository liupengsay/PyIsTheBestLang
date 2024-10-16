"""
Algorithm：bfs|deque_bfs|01-bfs|discretization_bfs|bound_bfs|coloring_method|odd_circle
Description：multi_source_bfs|bilateral_bfs|spfa|a-star|heuristic_search

====================================LeetCode====================================
1617（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）brute_force|tree_diameter
100318（https://leetcode.cn/problems/find-minimum-diameter-after-merging-two-trees）graph_diameter|diameter_merge|classical

=====================================LuoGu======================================
P1099（https://www.luogu.com.cn/problem/P1099）tree_diameter|bfs|two_pointers|monotonic_queue|classical|greedy
P2491（https://www.luogu.com.cn/problem/P2491）tree_diameter|bfs|two_pointers|monotonic_queue|classical|greedy
P3304（https://www.luogu.com.cn/problem/P3304）tree_diameter

===================================CodeForces===================================
1805D（https://codeforces.com/problemset/problem/1805/D）tree_diameter
455C（https://codeforces.com/problemset/problem/455/C）bfs|graph_diameter|union_find|implemention|diameter_merge
734E（https://codeforces.com/problemset/problem/734/E）tree_diameter|brain_teaser|greedy|shrink_node

====================================AtCoder=====================================
ABC267F（https://atcoder.jp/contests/abc267/tasks/abc267_f）tree_diameter|reroot_dp|brain_teaser|dfs|back_trace|classical
ABC221F（https://atcoder.jp/contests/abc221/tasks/abc221_f）tree_diameter|linear_dp
ABC361E（https://atcoder.jp/contests/abc361/tasks/abc361_e）tree_diameter|classical

=====================================AcWing=====================================

=====================================LibraryChcker=====================================
1（https://judge.yosupo.jp/problem/tree_diameter）tree_diameter

"""
from collections import Counter
from typing import List

from src.graph.tree_diameter.template import TreeDiameter, GraphDiameter
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


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
        u, v = tree.get_diameter_math.info()[:2]
        dis1, _ = tree.get_bfs_dis(u)
        dis2, _ = tree.get_bfs_dis(v)
        diff = [0] * (n + 1)
        for i in range(n):
            diff[max(dis1[i], dis2[i]) + 1] += 1
        diff[0] = 1
        for i in range(1, n + 1):
            diff[i] += diff[i - 1]
        ac.lst([min(x, n) for x in diff[1:]])
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
        dis = [math.inf] * n
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

        dis = [math.inf] * n
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
                cur = max(cur, d)
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
                cur = max(cur, d)
                for y, w in dct[x]:
                    if y != fa and dis[y] != -1:
                        stack.append((y, d + w, x))
            left = i
            if cur == tot - pre:
                break

        ans = max(0, right - left)
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
            ans[tree.get_diameter_math.info()[-1]] += 1
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
        _, _, path, d = tree.get_diameter_math.info()
        ac.lst([d, len(path)])
        ac.lst(path)
        return

    @staticmethod
    def abc_267f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc267/tasks/abc267_f
        tag: tree_diameter|reroot_dp|brain_teaser|dfs|back_trace|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append((j, 1))
            dct[j].append((i, 1))
        _, _, stack, _ = TreeDiameter(dct).get_diameter_math.info()
        ind = {num: i for i, num in enumerate(stack)}
        queries = [[] for _ in range(n)]
        q = ac.read_int()
        for i in range(q):
            u, k = ac.read_list_ints()
            queries[u - 1].append((k, i))
        visit = [0] * n
        for i in stack:
            visit[i] = 1
        ans = [-2] * q
        depth = [0] * n
        for i in stack:
            fa = ind[i]
            cur = [(i, 0)]
            while cur:
                x, d = cur.pop()
                depth[d] = x
                for k, y in queries[x]:
                    if k > d:
                        dd = k - d
                        if fa + dd < len(stack):
                            ans[y] = stack[fa + dd]
                        elif fa >= dd:
                            ans[y] = stack[fa - dd]
                    else:
                        ans[y] = depth[d - k]
                for y, _ in dct[x]:
                    if not visit[y]:
                        cur.append((y, d + 1))
                        visit[y] = 1
        for a in ans:
            ac.st(a + 1)
        return

    @staticmethod
    def abc_221f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc221/tasks/abc221_f
        tag: diameter|linear_dp
        """
        mod = 998244353
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append((j, 1))
            dct[j].append((i, 1))
        if n == 2:
            ac.st(1)
            return

        x, y, path, d = TreeDiameter(dct).get_diameter_math.info()
        if len(path) % 2 == 0:
            m = len(path)
            i, j = path[m // 2 - 1], path[m // 2]
            n += 1
            dct.append([])
            dct[i].remove((j, 1))
            dct[j].remove((i, 1))
            dct[n - 1].append((i, 1))
            dct[i].append((n - 1, 1))
            dct[n - 1].append((j, 1))
            dct[j].append((n - 1, 1))
            x, y, path, d = TreeDiameter(dct).get_diameter_math.info()
        m = len(path)
        mid = path[m // 2]
        dis = [-1] * n
        stack = [mid]
        dis[mid] = 0
        parent = [-1] * n

        while stack:
            nex = []
            for i in stack:
                for j, _ in dct[i]:
                    if dis[j] == -1:
                        if parent[i] == -1:
                            parent[j] = j
                        else:
                            parent[j] = parent[i]
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex[:]
        ceil = max(dis)

        cnt = list(Counter([parent[x] for x in range(n) if dis[x] == ceil]).values())
        dp = [1, 0, 0]
        for c in cnt:
            ndp = [0, 0, 0]
            ndp[0] = dp[0]
            ndp[1] = dp[0] * c + dp[1]
            ndp[2] = (dp[1] + dp[2]) * c + dp[2]
            dp = [x % mod for x in ndp]
        ac.st(dp[-1])
        return

    @staticmethod
    def cf_455c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/455/C
        tag: bfs|graph_diameter|union_find|implemention|diameter_merge
        """
        n, m, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i, j)
            dct[i].append(j)
            dct[j].append(i)
        diameter = [0] * n
        group = uf.get_root_part()
        for g in group:
            if len(group[g]) > 1:
                lst = group[g][:]
                m = len(lst)
                ind = {num: i for i, num in enumerate(lst)}
                edge = [[] for _ in range(m)]
                for i in lst:
                    for j in dct[i]:
                        edge[ind[i]].append(ind[j])
                        edge[ind[j]].append(ind[i])
                diameter[g] = GraphDiameter().get_diameter(edge)

        def check(aa, bb):
            if aa > bb:
                aa, bb = bb, aa
            return max((aa + 1) // 2 + 1, bb // 2) + (bb + 1) // 2

        for _ in range(q):
            lst = ac.read_list_ints_minus_one()
            if lst[0] == 0:
                ac.st(diameter[uf.find(lst[1])])
            else:
                x, y = lst[1:]
                a, b = diameter[uf.find(x)], diameter[uf.find(y)]
                if uf.union(x, y):
                    diameter[uf.find(x)] = check(a, b)
        return

    @staticmethod
    def lc_100318(edges1: List[List[int]], edges2: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/find-minimum-diameter-after-merging-two-trees
        tag: graph_diameter|diameter_merge|classical
        """
        n = len(edges1) + 1
        dct1 = [[] for _ in range(n)]
        for i, j in edges1:
            dct1[i].append(j)
            dct1[j].append(i)
        path1 = GraphDiameter().get_diameter(dct1, 0)

        m = len(edges2) + 1
        dct1 = [[] for _ in range(m)]
        for i, j in edges2:
            dct1[i].append(j)
            dct1[j].append(i)
        path2 = GraphDiameter().get_diameter(dct1, 0)

        def check(aa, bb):
            if aa > bb:
                aa, bb = bb, aa
            return max((aa + 1) // 2 + 1, bb // 2) + (bb + 1) // 2

        return check(path1, path2)

    @staticmethod
    def cf_734e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/734/E
        tag: tree_diameter|brain_teaser|greedy|shrink_node
        """
        n = ac.read_int()
        color = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            if color[i] == color[j]:
                dct[i].append((j, 0))
                dct[j].append((i, 0))
            else:
                dct[i].append((j, 1))
                dct[j].append((i, 1))
        _, _, _, dis = TreeDiameter(dct).get_diameter_math.info()
        ac.st((dis + 1) // 2)
        return