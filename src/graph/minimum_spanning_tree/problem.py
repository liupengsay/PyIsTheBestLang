"""

Algorithm：mst|kruskal|prim|strictly_second_mst|multiplication_method|lca|brute_force|shortest_path_spanning_tree|greedy
Description：prim is node wise and kruskal is edge wise, prim is suitable for dense graph and kruskal is suitable for sparse graph


====================================LeetCode====================================
1489（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）mst|necessary_edge|fake_necessary_edge
1584（https://leetcode.cn/problems/min-cost-to-connect-all-points/）manhattan_distance|dense_graph|prim|mst
1724（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths-ii/）mst|classical|multiplication_method|lca

=====================================LuoGu======================================
P3366（https://www.luogu.com.cn/problem/P3366）mst
P2820（https://www.luogu.com.cn/problem/P2820）reverse_thinking|mst
P1991（https://www.luogu.com.cn/problem/P1991）mst|connected_graph
P1661（https://www.luogu.com.cn/problem/P1661）mst
P1547（https://www.luogu.com.cn/problem/P1547）mst
P2121（https://www.luogu.com.cn/problem/P2121）limited_mst
P2126（https://www.luogu.com.cn/problem/P2126）mst
P2872（https://www.luogu.com.cn/problem/P2872）prim|mst
P2330（https://www.luogu.com.cn/problem/P2330）mst
P2504（https://www.luogu.com.cn/problem/P2504）mst
P2700（https://www.luogu.com.cn/problem/P2700）reverse_thinking|mst|union_find|size
list?user=739032&status=12&page=13（https://www.luogu.com.cn/record/list?user=739032&status=12&page=13）mst
P1194（https://www.luogu.com.cn/problem/P1194）mst
P2916（https://www.luogu.com.cn/problem/P2916）custom_sort|mst|classical
P4955（https://www.luogu.com.cn/problem/P4955）mst
P6705（https://www.luogu.com.cn/problem/P6705）brute_force|mst|union_find|uf_initial
P7775（https://www.luogu.com.cn/problem/P7775）bfs|mst_like
P2658（https://www.luogu.com.cn/problem/P2658）classical|mst
P4180（https://www.luogu.com.cn/problem/P4180）mst|lca|multiplication_method|strictly_second_mst|classical
P1265（https://www.luogu.com.cn/problem/P1265）prim|mst
P1340（https://www.luogu.com.cn/problem/P1340）reverse_order|union_find|mst
P2212（https://www.luogu.com.cn/problem/P2212）prim|mst|dense_graph
P2847（https://www.luogu.com.cn/problem/P2847）prim|mst|dense_graph
P3535（https://www.luogu.com.cn/problem/P3535）mst|judge_circle_by_union_find|connected
P4047（https://www.luogu.com.cn/problem/P4047）mst
P6171（https://www.luogu.com.cn/problem/P6171）sparse|kruskal|mst
P1550（https://www.luogu.com.cn/problem/P1550）mst|build_graph|fake_source|classical
P1661（https://www.luogu.com.cn/problem/P1661）manhattan_distance|mst|classical

===================================CodeForces===================================
609E（https://codeforces.com/problemset/problem/609/E）lca|greedy|mst|strictly_second_mst|necessary_edge
1108F（https://codeforces.com/contest/1108/problem/F）mst|classical
1095F（https://codeforces.com/contest/1095/problem/F）mst|brain_teaser|greedy
1624G（https://codeforces.com/contest/1624/problem/G）or_mst|classical
1857G（https://codeforces.com/contest/1857/problem/G）mst|brain_teaser|classical
1981E（https://codeforces.com/contest/1981/problem/E）scan_line|union_find|classical|implemention|mst
1242B（https://codeforces.com/contest/1242/problem/B）mst|observation|prim|brain_teaser|implemention|bfs
1245D（https://codeforces.com/problemset/problem/1245/D）prim|specific_plan|virtual_source|classical
1513D（https://codeforces.com/problemset/problem/1513/D）bfs|gcd_like|mst
888G（https://codeforces.com/contest/888/problem/G）heuristic_merge|get_minimum_xor|mst|classical
587C（https://codeforces.com/problemset/problem/587/C）tree_multiplication_method|multiplication_method|classical

====================================AtCoder=====================================
ARC076B（https://atcoder.jp/contests/abc065/tasks/arc076_b）mst
ABC282E（https://atcoder.jp/contests/abc282/tasks/abc282_e）union_find|mst|brain_teaser|classical
ABC235E（https://atcoder.jp/contests/abc235/tasks/abc235_e）online_query|data_range|offline_query|mst|maximum_weight|classical
ABC352E（https://atcoder.jp/contests/abc352/tasks/abc352_e）union_find|mst|reverse_order
ABC181F（https://atcoder.jp/contests/abc181/tasks/abc181_f）geometry|binary_search|mst|classical

=====================================AcWing=====================================
3731（https://www.acwing.com/problem/content/3731/）prim|mst|dense_graph|specific_plan

================================LibraryChecker================================
1（https://judge.yosupo.jp/problem/manhattanmst）manhattan_distance|mst|classical
Directed MST（https://judge.yosupo.jp/problem/directedmst）
3（https://codeforces.com/edu/course/2/lesson/7/2/practice/contest/289391/problem/F）mst|brute_force
4（https://codeforces.com/edu/course/2/lesson/7/2/practice/contest/289391/problem/H）mst|greedy
5（https://vjudge.net/problem/BZOJ-2177）manhattan_distance|mst|classical
6（https://www.51nod.com/Challenge/Problem.html#problemId=1213）manhattan_distance|mst|classical

"""
import math
from collections import defaultdict
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.graph.minimum_spanning_tree.template import KruskalMinimumSpanningTree, \
    PrimMinimumSpanningTree, ManhattanMST, TreeAncestorMinIds, \
    TreeMultiplicationMaxWeights, TreeMultiplicationMaxSecondWeights
from src.graph.tarjan.template import Tarjan
from src.graph.union_find.template import UnionFind, PersistentUnionFind
from src.struct.sorted_list.template import SortedList
from src.struct.trie_like.template import BinaryTrieXor
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1991(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1991
        tag: mst|connected_grap|classical|kruskal
        """
        k, n = ac.read_list_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            for j in range(i + 1, n):
                a = pos[i][0] - pos[j][0]
                b = pos[i][1] - pos[j][1]
                edge.append((i, j, a * a + b * b))

        uf = UnionFind(n)
        edge.sort(key=lambda it: it[2])
        cost = 0
        for x, y, z in edge:
            if uf.part == k:
                break
            if uf.union(x, y):
                cost = z
        ans = math.sqrt(cost)
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p2820(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2820
        tag: reverse_thinking|mst
        """
        n, m = ac.read_list_ints()
        edge = [ac.read_list_ints() for _ in range(m)]
        uf = UnionFind(n)
        edge.sort(key=lambda it: it[2])
        cost = 0
        for x, y, z in edge:
            if not uf.union(x - 1, y - 1):
                cost += z
        ac.st(cost)
        return

    @staticmethod
    def lg_p3366_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3366
        tag: mst
        """
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = KruskalMinimumSpanningTree(edges, n, "kruskal")
        if mst.cost == -1:
            ac.st("orz")
        else:
            ac.st(mst.cost)
        return

    @staticmethod
    def lg_p3366_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3366
        tag: mst
        """
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = KruskalMinimumSpanningTree(edges, n, "prim")
        if mst.cnt < n:
            ac.st("orz")
        else:
            ac.st(mst.cost)
        return

    @staticmethod
    def cf_1108f_1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1108/problem/F
        tag: mst|second_mst|multiplication_method|union_find
        """
        n, m = ac.read_list_ints()
        weights = []
        fro = []
        to = []
        use = [0] * m
        for x in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            weights.append(w * m + x)
            fro.append(i)
            to.append(j)
        weights.sort()
        uf = UnionFind(n)
        graph = TreeMultiplicationMaxWeights(n)
        cost = 0
        for val in weights:
            w, x = val // m, val % m
            i, j = fro[x], to[x]
            if uf.union(i, j):
                cost += w
                use[x] = 1
                graph.add_undirected_edge(i, j, w)
            if uf.part == 1:
                break
        graph.build_multiplication()
        ans = 0
        for val in weights:
            w, x = val // m, val % m
            if not use[x]:
                i, j = fro[x], to[x]
                dis = graph.get_max_weights_between_nodes(i, j)
                if dis == w:
                    ans += 1
        ac.st(ans)
        return

    @staticmethod
    def cf_1108f_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1108/problem/F
        tag: mst|second|multiplication_method|union_find|classical|hard
        """
        n, m = ac.read_list_ints()
        dct = defaultdict(list)
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[w].append((i - 1, j - 1))
        uf = UnionFind(n)
        ans = 0
        for w in sorted(dct):
            for i, j in dct[w]:
                if not uf.is_connected(i, j):
                    ans += 1
            for i, j in dct[w]:
                if uf.union(i, j):
                    ans -= 1
        ac.st(ans)
        return

    @staticmethod
    def lc_1489(n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
        tag: mst|necessary_edge|fake_necessary_edge|brain_teaser|classical|tarjan|cutting_edge|hard
        """

        m = len(edges)
        dct = defaultdict(list)
        for i in range(m):
            dct[edges[i][2]].append(i)

        uf = UnionFind(n)
        key = []
        fake = []
        for w in sorted(dct):
            cur_dct = defaultdict(list)
            nodes = set()
            for i in dct[w]:
                x, y, _ = edges[i]
                rx, ry = uf.find(x), uf.find(y)
                if rx != ry:
                    if rx > ry:
                        rx, ry = ry, rx
                    nodes.add(rx)
                    nodes.add(ry)
                    cur_dct[(rx, ry)].append(i)
            nodes = sorted(nodes)
            ind = {num: i for i, num in enumerate(nodes)}
            cur_edge = [[] for _ in range(len(nodes))]
            for rx, ry in cur_dct:
                cur_edge[ind[rx]].append(ind[ry])
                cur_edge[ind[ry]].append(ind[rx])
            _, cutting_edge = Tarjan().get_cut(len(nodes), cur_edge)

            for rx, ry in cur_dct:
                if len(cur_dct[(rx, ry)]) > 1:
                    fake.extend(cur_dct[(rx, ry)])
                elif (ind[rx], ind[ry]) in cutting_edge:
                    key.append(cur_dct[(rx, ry)][0])
                else:
                    fake.append(cur_dct[(rx, ry)][0])
                uf.union(rx, ry)

        return [key, fake]

    @staticmethod
    def lg_p2872(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2872
        tag: prim|mst|classical
        """

        def dis(x1, y1, x2, y2):
            res = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            return res ** 0.5

        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for i in range(m):
            u, v = ac.read_list_ints_minus_one()
            dct[u][v] = dct[v][u] = 0

        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            x, y = nums[i]
            nex = -1
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                visit[j] = min(visit[j], dct[i].get(j, math.inf))
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1194(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1194
        tag: mst|classical|build_graph
        """

        a, b = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(b)]
        edge = [(0, i, a) for i in range(1, b + 1)]
        for i in range(b):
            for j in range(i + 1, b):
                if 0 < grid[i][j] < a:
                    edge.append((i + 1, j + 1, grid[i][j]))
        mst = KruskalMinimumSpanningTree(edge, b + 1)
        ac.st(mst.cost)
        return

    @staticmethod
    def lg_p4180(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4180
        tag: mst|lca|multiplication_method|strictly_second_mst|classical
        """
        n, m = ac.read_list_ints()  # TLE
        weights = []
        fro = []
        to = []
        for x in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            weights.append(w * m + x)
            fro.append(i)
            to.append(j)
        weights.sort()
        uf = UnionFind(n)
        graph = TreeMultiplicationMaxSecondWeights(n)
        cost = 0
        for val in weights:
            w, x = val // m, val % m
            i, j = fro[x], to[x]
            if uf.union(i, j):
                cost += w
                graph.add_undirected_edge(i, j, w)
            if uf.part == 1:
                break
        graph.build_multiplication()
        ans = 10 ** 14
        for val in weights:
            w, x = val // m, val % m
            i, j = fro[x], to[x]
            dis = graph.get_max_weights_between_nodes(i, j)
            for weight in dis:
                if -1 < weight < w:
                    cur = cost - weight + w
                    ans = min(cur, ans)
        ac.st(ans)
        return

    @staticmethod
    def cf_609e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/609/E
        tag: lca|greedy|mst|strictly_second_mst|necessary_edge
        """
        """
        url: https://codeforces.com/problemset/problem/609/E
        tag: lca|greedy|mst|tree_multiplication|maximum_weight|second_mst|strictly_second_minimum_spanning_tree
        """
        n, m = ac.read_list_ints()
        weights = []
        fro = []
        to = []
        for x in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            weights.append(w * m + x)
            fro.append(i)
            to.append(j)
        weights.sort()
        uf = UnionFind(n)
        graph = TreeMultiplicationMaxWeights(n)
        cost = 0
        ans = [-1] * m
        use = []
        for val in weights:
            w, x = val // m, val % m
            i, j = fro[x], to[x]
            if uf.union(i, j):
                cost += w
                graph.add_undirected_edge(i, j, w)
                use.append(x)
            if uf.part == 1:
                break
        graph.build_multiplication()
        for x in use:
            ans[x] = cost
        for val in weights:
            w, x = val // m, val % m
            if ans[x] == -1:
                i, j = fro[x], to[x]
                dis = graph.get_max_weights_between_nodes(i, j)
                ans[x] = cost - dis + w
        ac.flatten(ans)
        return

    @staticmethod
    def lg_p1265(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1265
        tag: prim|mst
        """

        def dis(x1, y1, x2, y2):
            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d ** 0.5
            nex = -1
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1340_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1340
        tag: reverse_order|union_find|mst|reverse_thinking
        """

        n, w = ac.read_list_ints()

        edges = [ac.read_list_ints() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: edges[it][-1])

        uf = UnionFind(n)
        ans = []
        select = set()
        cost = 0
        for i in range(w - 1, -1, -1):
            if uf.part > 1:
                cost = 0
                select = set()
                for j in ind:
                    if j <= i:
                        x, y, ww = edges[j]
                        if uf.union(x - 1, y - 1):
                            cost += ww
                            select.add(j)
            if uf.part > 1:
                ans.append(-1)
                break
            ans.append(cost)
            if i in select:
                uf.initialize()
                select = set()
                cost = 0
        while len(ans) < w:
            ans.append(-1)

        for i in range(w - 1, -1, -1):
            ls = ans[i]
            ac.st(ls)
        return

    @staticmethod
    def lg_p1340_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1340
        tag: reverse_order|union_find|mst|reverse_thinking
        """
        n, m = ac.read_list_ints()
        uf = UnionFind(n)
        lst = SortedList()
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            uf.union(i - 1, j - 1)
            lst.add((w, i - 1, j - 1))
            if uf.part > 1:
                ac.st(-1)
            else:
                uf.initialize()
                cost = 0
                for w, i, j in lst:
                    if uf.union(i, j):
                        cost += w
                    if uf.part == 1:
                        break
                ac.st(cost)
        return

    @staticmethod
    def lg_p1550(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1550
        tag: mst|build_graph|fake_source|classical
        """

        n = ac.read_int()
        edges = []
        for i in range(n):
            w = ac.read_int()
            edges.append((0, i + 1, w))

        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i + 1, n):
                edges.append((i + 1, j + 1, grid[j]))
        edges.sort(key=lambda it: it[2])
        cost = 0
        uf = UnionFind(n + 1)
        for i, j, c in edges:
            if uf.union(i, j):
                cost += c
            if uf.part == 1:
                break
        ac.st(cost)
        return

    @staticmethod
    def lg_p2212(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2212
        tag: prim|mst|dense_graph
        """

        def dis(x1, y1, x2, y2):
            res = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            return res if res >= c else math.inf

        n, c = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        tree = PrimMinimumSpanningTree(dis)
        ac.st(tree.build(nums))
        return

    @staticmethod
    def lg_p2658(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2658
        tag: classical|mst
        """

        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        edge = []
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    edge.append((i * n + j, i * n + j + n, abs(grid[i][j] - grid[i + 1][j])))
                if j + 1 < n:
                    edge.append((i * n + j, i * n + j + 1, abs(grid[i][j] - grid[i][j + 1])))
        del grid

        uf = UnionFind(m * n)
        road = [0] * (m * n)
        s = 0
        for i in range(m):
            lst = ac.read_list_ints()
            for j in range(n):
                if lst[j]:
                    s += 1
                    road[i * n + j] = 1
        if s == 1:
            ac.st(0)
            return
        edge.sort(key=lambda it: it[2])
        for x, y, d in edge:
            a, b = road[uf.find(x)], road[uf.find(y)]
            if uf.union(x, y):
                road[uf.find(x)] = a + b
            if road[uf.find(x)] == s:
                ac.st(d)
                return
        return

    @staticmethod
    def lg_p2847(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2847
        tag: prim|mst|dense_graph
        """

        def dis(x1, y1, x2, y2):
            res = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            return res

        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans = max(ans, d)
            nex = -1
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p3535(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3535
        tag: mst|judge_circle_by_union_find|connected|reverse_thinking
        """

        n, m, k = ac.read_list_ints()
        edge = []
        uf = UnionFind(n)
        ans = 0
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if i >= k and j >= k:
                uf.union(i, j)
            else:
                edge.append((i, j))
        for i, j in edge:
            if not uf.union(i, j):
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p4047(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4047
        tag: mst
        """

        def dis():
            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            x1, y1 = nums[i]
            for j in range(i + 1, n):
                x2, y2 = nums[j]
                edge.append((i, j, dis()))
        edge.sort(key=lambda it: it[2])

        uf = UnionFind(n)
        ans = 0
        for i, j, d in edge:
            if uf.part == k:
                if not uf.is_connected(i, j):
                    ans = d
                    break
            else:
                uf.union(i, j)
        ans = ans ** 0.5
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p6171(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6171
        tag: sparse|kruskal|mst
        """
        a, b, n, m = ac.read_list_ints()
        nums1 = [0, a] + [ac.read_int() for _ in range(n)]
        nums2 = [0, b] + [ac.read_int() for _ in range(m)]
        nums1.sort()
        nums2.sort()
        dct = defaultdict(list)
        k = (m + 2) * (n + 2)
        for i in range(1, m + 2):
            for j in range(1, n + 2):
                x = (i - 1) * (n + 1) + (j - 1)
                if i + 1 < m + 2:
                    y = i * (n + 1) + (j - 1)
                    dct[nums1[j] - nums1[j - 1]].append(x * k + y)
                if j + 1 < n + 2:
                    y = (i - 1) * (n + 1) + j
                    dct[nums2[i] - nums2[i - 1]].append(x * k + y)
        del nums1
        del nums2
        ans = 0
        uf = UnionFind((n + 1) * (m + 1))
        for c in sorted(dct):
            for num in dct[c]:
                if uf.union(num // k, num % k):
                    ans += c
        ac.st(ans)
        return

    @staticmethod
    def lc_1584_1(points: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/min-cost-to-connect-all-points/
        tag: dense_graph|prim|mst
        """

        def dis(x1, y1, x2, y2):
            res = abs(x1 - x2) + abs(y1 - y2)
            return res

        tree = PrimMinimumSpanningTree(dis)
        return tree.build(points)

    @staticmethod
    def lc_1584_2(points: List[List[int]]) -> int:
        """
        url: manhattan_distance|dense_graph|prim|mst
        tag: dense_graph|prim|mst
        """
        return ManhattanMST().build(points)[0]

    @staticmethod
    def abc_076b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc065/tasks/arc076_b
        tag: mst|classical|brain_teaser
        """

        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it][0])
        edges = []
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][0] - nums[x][0]
            edges.append((x, y, d))

        uf = UnionFind(n)
        ind.sort(key=lambda it: nums[it][1])
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][1] - nums[x][1]
            edges.append((x, y, d))

        edges.sort(key=lambda it: it[2])
        ans = 0
        for i, j, d in edges:
            if uf.union(i, j):
                ans += d
        ac.st(ans)
        return

    @staticmethod
    def ac_3731(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3731/
        tag: prim|mst|dense_graph|specific_plan|classical|fake_source
        """

        def dis(aa, bb):
            if aa == 0:
                return cost[bb]
            if bb == 0:
                return cost[aa]
            return (k[aa] + k[bb]) * (abs(nums[aa][0] - nums[bb][0]) + abs(nums[aa][1] - nums[bb][1]))

        n = ac.read_int()
        nums = [[math.inf, math.inf]] + [ac.read_list_ints() for _ in range(n)]
        cost = [math.inf] + ac.read_list_ints()
        k = [math.inf] + ac.read_list_ints()

        ans = nex = 0
        rest = set(list(range(n + 1)))
        visit = [math.inf] * (n + 1)
        visit[nex] = 0
        pre = [-1] * (n + 1)
        edge = []
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            for j in rest:
                dj = dis(i, j)
                if dj < visit[j]:
                    visit[j] = dj
                    pre[j] = i
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
            if nex != -1:
                edge.append((pre[nex], nex))
        ac.st(ans if ans < math.inf else -1)

        lst = []
        for a, b in edge:
            if a == 0:
                lst.append(b)
            elif b == 0:
                lst.append(a)
        ac.st(len(lst))
        ac.lst(lst)
        ac.st(len(edge) - len(lst))
        for a, b in edge:
            if a and b:
                ac.lst([a, b])
        return

    @staticmethod
    def lc_1724_1():
        """
        url: https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths-ii/
        tag: mst|classical|multiplication_method|lca
        """

        class DistanceLimitedPathsExist:
            def __init__(self, n: int, edge_list: List[List[int]]):
                uf = UnionFind(n + 1)
                edge_list.extend([[n, i, math.inf] for i in range(n)])
                dct = [dict() for _ in range(n + 1)]
                for i, j, d in sorted(edge_list, key=lambda it: it[-1]):
                    if uf.union(i, j):
                        dct[i][j] = dct[j][i] = d
                # self.tree = SecondMinimumSpanningTreeLight(dct)
                self.tree = TreeMultiplicationMaxWeights(n)

            def query(self, p: int, q: int, limit: int) -> bool:
                maximum = self.tree.get_max_weights_between_nodes(p, q)
                return maximum < limit

        return DistanceLimitedPathsExist

    @staticmethod
    def lc_1724_2():
        """
        url: https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths-ii/
        tag: mst|classical|multiplication_method|lca
        """

        class DistanceLimitedPathsExist:

            def __init__(self, n: int, edge_list: List[List[int]]):
                self.puf = PersistentUnionFind(n)
                edge_list.sort(key=lambda it: it[2])
                for x, y, tm in edge_list:
                    self.puf.union(x, y, tm)
                return

            def query(self, p: int, q: int, limit: int) -> bool:
                return self.puf.is_connected(p, q, limit)

        return DistanceLimitedPathsExist

    @staticmethod
    def library_check_3(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/7/2/practice/contest/289391/problem/F
        tag: mst|brute_force
        """
        n, m = ac.read_list_ints()
        edges = [tuple(ac.read_list_ints()) for _ in range(m)]
        edges.sort(key=lambda it: it[2])
        uf = UnionFind(n)
        for x in range(m - 1, -1, -1):
            i, j, w = edges[x]
            uf.union(i - 1, j - 1)
            if uf.part == 1:
                break
        else:
            ac.no()
            return
        ans = math.inf
        uf = UnionFind(n)
        for i in range(x + 1):
            uf.initialize()
            s = edges[i][2]
            for j in range(i, m):
                x, y, w = edges[j]
                uf.union(x - 1, y - 1)
                if uf.part == 1 or w - s > ans:
                    if uf.part == 1 and w - s <= ans:
                        ans = w - s
                    break
        ac.yes()
        ac.st(ans)
        return

    @staticmethod
    def library_check_4(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/7/2/practice/contest/289391/problem/H
        tag: mst|greedy
        """
        n, m, s = ac.read_list_ints()
        edges = [tuple(ac.read_list_ints()) for _ in range(m)]
        ind = list(range(m))
        ind.sort(key=lambda it: -edges[it][2])
        uf = UnionFind(n)
        rest = []
        cost = 0
        for i in ind:
            x, y, w = edges[i]
            if not uf.union(x - 1, y - 1):
                rest.append(i)
                cost += w
        rest.reverse()
        while cost > s:
            cost -= edges[rest.pop()][2]
        ac.st(len(rest))
        ac.lst([x + 1 for x in rest])
        return

    @staticmethod
    def abc_282e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc282/tasks/abc282_e
        tag: union_find|mst|brain_teaser|classical
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = []
        for i in range(n):
            for j in range(i + 1, n):
                x, y = nums[j], nums[i]
                score = (pow(x, y, m) + pow(y, x, m)) % m
                lst.append((score, i, j))
        lst.sort(reverse=True)
        ans = 0
        uf = UnionFind(n)
        for score, i, j in lst:
            if uf.union(i, j):
                ans += score
        ac.st(ans)
        return

    @staticmethod
    def library_checker_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/manhattanmst
        tag: manhattan_distance|mst|classical
        """
        n = ac.read_int()  # TLE
        ans = ManhattanMST().build([ac.read_list_ints() for _ in range(n)])
        ac.st(ans[0])
        for ls in ans[1]:
            ac.lst(ls)
        return

    @staticmethod
    def library_checker_5(ac=FastIO()):
        """
        url: https://vjudge.net/problem/BZOJ-2177
        tag: manhattan_distance|mst|classical
        """
        n = ac.read_int()
        ans = ManhattanMST().build([ac.read_list_ints() for _ in range(n)])
        ac.st(ans[0])
        return

    @staticmethod
    def library_checker_6(ac=FastIO()):
        """
        url: https://www.51nod.com/Challenge/Problem.html#problemId=1213
        tag: manhattan_distance|mst|classical
        """
        n = ac.read_int()
        ans = ManhattanMST().build([ac.read_list_ints() for _ in range(n)])
        ac.st(ans[0])
        return

    @staticmethod
    def lg_p1661(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1661
        tag: manhattan_distance|mst|classical
        """
        n = ac.read_int()
        ans = ManhattanMST().build([ac.read_list_ints() for _ in range(n)])
        ac.st((ans[-1][-1] + 1) // 2)
        return

    @staticmethod
    def abc_235e_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc235/tasks/abc235_e
        tag: online_query|data_range|offline_query|mst|maximum_weight|classical
        """
        n, m, q = ac.read_list_ints()
        weights = []
        fro = []
        to = []
        for x in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            weights.append(w * m + x)
            fro.append(i)
            to.append(j)
        weights.sort()
        uf = UnionFind(n)
        graph = TreeMultiplicationMaxWeights(n)
        cost = 0
        for val in weights:
            w, x = val // m, val % m
            i, j = fro[x], to[x]
            if uf.union(i, j):
                cost += w
                graph.add_undirected_edge(i, j, w)
            if uf.part == 1:
                break
        graph.build_multiplication()

        for _ in range(q):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            dis = graph.get_max_weights_between_nodes(i, j)
            if dis > w:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def abc_235e_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc235/tasks/abc235_e
        tag: online_query|data_range|offline_query|mst|maximum_weight|classical
        """
        n, m, q = ac.read_list_ints()
        edges = []
        for i in range(m):
            lst = ac.read_list_ints() + [i]
            edges.append(lst)
        for i in range(q):
            lst = ac.read_list_ints() + [i + m]
            edges.append(lst)
        edges.sort(key=lambda it: it[-2])
        uf = UnionFind(n + 1)
        visit = [0] * (m + q)
        for a, b, _, i in edges:
            if not uf.is_connected(a, b):
                visit[i] = 1
            if i < m:
                uf.union(a, b)
        for i in range(m, m + q):
            if visit[i]:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def cf_1981e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1981/problem/E
        tag: scan_line|union_find|classical|implemention|mst
        """
        for _ in range(ac.read_int()):  # TLE
            n = ac.read_int()
            nums = [ac.read_list_ints() for _ in range(n)]
            nodes = set()
            for x, y, _ in nums:
                nodes.add(x)
                nodes.add(y)
            ac.get_random_seed()
            start = defaultdict(list)
            end = defaultdict(list)
            for i, (x, y, _) in enumerate(nums):
                start[x ^ ac.random_seed].append(i)
                end[y ^ ac.random_seed].append(i)
            lst = SortedList()
            edges = []
            for x in sorted(nodes):
                for i in start[x ^ ac.random_seed]:
                    lst.add((nums[i][2], i))

                for i in start[x ^ ac.random_seed]:
                    lst.discard((nums[i][2], i))
                    j = lst.bisect_left((nums[i][2], -1))
                    if j - 1 >= 0:
                        edges.append((i, lst[j - 1][1], abs(nums[i][2] - nums[lst[j - 1][1]][2])))
                    if j < len(lst):
                        edges.append((i, lst[j][1], abs(nums[i][2] - nums[lst[j][1]][2])))
                    lst.add((nums[i][2], i))

                for i in end[x ^ ac.random_seed]:
                    lst.discard((nums[i][2], i))
            uf = UnionFind(n)
            edges.sort(key=lambda it: it[2])
            ans = 0
            for i, j, w in edges:
                if uf.union(i, j):
                    ans += w
            ac.st(ans if uf.part == 1 else -1)
        return

    @staticmethod
    def cf_1242b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1242/problem/B
        tag: mst|observation|prim|brain_teaser|implemention|bfs
        """
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if i > j:
                i, j = j, i
            dct[i].add(j)
            dct[j].add(i)

        ans = 0
        zero = []
        rest = set(range(n))
        while rest:
            if zero:
                root = zero.pop()
            else:
                root = rest.pop()
                ans += 1
            cur = {x for x in rest if root not in dct[x]}
            zero += cur
            rest -= cur
        ac.st(ans - 1)
        return

    @staticmethod
    def cf_1245d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1245/D
        tag: prim|specific_plan|virtual_source|classical
        """
        n = ac.read_int()
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)]
        c = [0] + ac.read_list_ints()
        k = [0] + ac.read_list_ints()

        def dis(ii, jj):
            if ii == 0:
                return c[jj]
            if jj == 0:
                return c[ii]
            return (k[ii] + k[jj]) * (abs(nums[ii][0] - nums[jj][0]) + abs(nums[ii][1] - nums[jj][1]))

        n = len(nums)
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        parent = [-1] * n
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            for j in rest:
                dj = dis(i, j)
                if dj < visit[j]:
                    visit[j] = dj
                    parent[j] = i
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        ac.st(ans)
        power = [i for i in range(n) if parent[i] == 0]
        ac.st(len(power))
        ac.lst(power)
        edges = [(i, parent[i]) for i in range(n) if parent[i] not in [0, -1]]
        ac.st(len(edges))
        for ls in edges:
            ac.lst(ls)
        return

    @staticmethod
    def cf_888g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/888/problem/G
        tag: heuristic_merge|get_minimum_xor|mst|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        uf = UnionFind(n)
        pre = dict()
        ac.get_random_seed()
        trie = BinaryTrieXor(1 << 30, n)
        for i in range(n):
            cur = nums[i] ^ ac.random_seed
            if cur in pre:
                uf.union(i, pre[cur])
            pre[cur] = i
            trie.add(nums[i], 1)
        ans = 0
        while uf.part > 1:
            group = uf.get_root_part()
            for g in group:
                for x in group[g]:
                    trie.remove(nums[x], 1)
                res = [math.inf, 0, 1]
                for x in group[g]:
                    y = trie.get_minimum_xor(nums[x])
                    if y < res[0]:
                        res = [y, x, pre[y ^ nums[x] ^ ac.random_seed]]
                if uf.union(res[1], res[2]):
                    ans += res[0]
                for x in group[g]:
                    trie.add(nums[x], 1)
        ac.st(ans)
        return

    @staticmethod
    def cf_587c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/587/C
        tag: tree_multiplication_method|multiplication_method|classical
        """
        n, m, q = ac.read_list_ints()  # math.inf making tle !!
        tree = TreeAncestorMinIds(n)
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            tree.add_undirected_edge(u, v)

        ids = [n + 1] * 10 * n
        ind = [0] * n
        c = ac.read_list_ints_minus_one()
        for i in range(m):
            if ind[c[i]] < 10:
                ids[c[i] * 10 + ind[c[i]]] = i + 1
                ind[c[i]] += 1
        tree.build_multiplication(ids)

        for _ in range(q):
            u, v, a = ac.read_list_ints_minus_one()
            a += 1
            ans = tree.get_min_ids_between_nodes(u, v)[:a]
            ans = [x for x in ans if x < n + 1]
            ac.lst([len(ans)] + ans)
        return

    @staticmethod
    def abc_181f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc181/tasks/abc181_f
        tag: geometry|binary_search|mst|classical
        """
        n = ac.read_int()
        nums = []
        for _ in range(n):
            x, y = ac.read_list_ints()
            nums.append((x, y))
            nums.append((x, -100))
            nums.append((x, 100))

        def check(d):
            uf = UnionFind(len(nums))
            for i in range(len(nums)):
                aa, bb = nums[i]
                for j in range(i + 1, len(nums)):
                    cc, dd = nums[j]
                    if (aa - cc) ** 2 + (bb - dd) ** 2 <= 4 * d * d:
                        uf.union(i, j)
            group = uf.get_root_part()
            for g in group:
                lst = [nums[i][1] for i in group[g]]
                if -100 in lst and 100 in lst:
                    return False
            return True

        ans = BinarySearch().find_float_right(0, 100, check, 1e-9)
        ac.st(ans)
        return
