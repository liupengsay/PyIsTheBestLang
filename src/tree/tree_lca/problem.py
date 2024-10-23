"""

Algorithm：lca|multiplication_method|tree_chain_split|offline_lca|online_lca
Description：

====================================LeetCode====================================
1483（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）sparse_table|tree_array|lca|tree_lca|classical
2846（https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree）tree_lca|greed


=====================================LuoGu======================================
P3379（https://www.luogu.com.cn/problem/P3379）tree_lca|classical
P7128（https://www.luogu.com.cn/problem/P7128）lca|implemention|sort
P7167（https://www.luogu.com.cn/problem/P7167）monotonic_stack|tree_lca|build_tree
P2912（https://www.luogu.com.cn/problem/P2912）offline_lca|offline_query
P3019（https://www.luogu.com.cn/problem/P3019）offline_query|lca
P3384（https://www.luogu.com.cn/problem/P3384）tree_chain_split|tree_array|implemention
P3976（https://www.luogu.com.cn/problem/P3976）range_add|range_max_gain|range_min_gain

======================================LibraryChecker==================================
1（https://judge.yosupo.jp/problem/lca）tree_lca

===================================CodeForces===================================
1328E（https://codeforces.com/problemset/problem/1328/E）tree_lca|dfs_order
321C（https://codeforces.com/problemset/problem/321/C）tree_centroid_recursion|classical
519E（https://codeforces.com/problemset/problem/519/E）lca|kth_ancestor|counter
1296F（https://codeforces.com/contest/1296/problem/F）offline_lca|greed|construction|multiplication_method
1702G2（https://codeforces.com/contest/1702/problem/G2）tree_lca
1843F2（https://codeforces.com/contest/1843/problem/F2）tree_lca|multiplication_method|classical|max_con_sub_sum
1304E（https://codeforces.com/problemset/problem/1304/E）observation|tree_lca|graph|implemention
187E（https://atcoder.jp/contests/abc187/tasks/abc187_e）tree_diff_array|tree_dfs_order|tree_lca

====================================AtCoder=====================================
ABC294G（https://atcoder.jp/contests/abc294/tasks/abc294_g）segment_tree|point_set|range_sum|heavy_chain|tree_lca
ABC209D（https://atcoder.jp/contests/abc209/tasks/abc209_d）tree_ancestor
ABC202E（https://atcoder.jp/contests/abc202/tasks/abc202_e）heuristic_merge|offline_query|classical

=====================================AcWing=====================================
4202（https://www.acwing.com/problem/content/4205/）bit_operation|build_graph|tree_lca|tree_dis

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/BITTREEMIN）data_range|minimum_xor|tree_lca


"""
import math
from typing import List

from src.structure.segment_tree.template import PointSetRangeSum, RangeSetPointGet, RangeAddRangeMaxGainMinGain
from src.structure.tree_array.template import RangeAddRangeSum
from src.tree.tree_dp.template import WeightedTree
from src.tree.tree_lca.template import OfflineLCA, TreeCentroid, HeavyChain, TreeAncestorPool, \
    UnionFindGetLCA, TreeAncestorMaxSub
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p7167(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7167
        tag: monotonic_stack|tree_lca|build_tree|multiplication_method|classical|hard
        """
        n, q = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        parent = [n] * n
        edge = [[] for _ in range(n + 1)]
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]][0] < nums[i][0]:
                parent[stack.pop()] = i
            stack.append(i)
        for i in range(n):
            edge[n - parent[i]].append(n - i)

        weight = [x for _, x in nums] + [math.inf]
        tree = TreeAncestorPool(edge, weight[::-1])

        for _ in range(q):
            r, v = ac.read_list_ints()
            ans = tree.get_final_ancestor(n - r + 1, v)
            ac.st(0 if ans == 0 else n - ans + 1)
        return

    @staticmethod
    def cf_519e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/519/E
        tag: lca|kth_ancestor|counter|dis|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                self.parent = [-1] * self.n
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                sub[u] += sub[v]
                return

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        graph.lca_build_with_multiplication()
        sub = [1] * n
        graph.tree_dp()
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            if x == y:
                ac.st(n)
                continue
            dis = graph.lca_get_lca_and_dist_between_nodes(x, y)[1]
            if dis % 2 == 1:
                ac.st(0)
            else:
                z = graph.lca_get_lca_between_nodes(x, y)
                dis1 = graph.lca_get_lca_and_dist_between_nodes(x, z)[1]
                dis2 = graph.lca_get_lca_and_dist_between_nodes(y, z)[1]
                if dis1 == dis2:
                    up = n - sub[z]
                    down = sub[z] - sub[graph.lca_get_kth_ancestor(x, dis1 - 1)] - sub[
                        graph.lca_get_kth_ancestor(y, dis2 - 1)]
                    ac.st(up + down)
                elif dis1 > dis2:
                    w = graph.lca_get_kth_ancestor(x, (dis1 + dis2) // 2)
                    ac.st(sub[w] - sub[graph.lca_get_kth_ancestor(x, (dis1 + dis2) // 2 - 1)])
                else:
                    w = graph.lca_get_kth_ancestor(y, (dis1 + dis2) // 2)
                    ac.st(sub[w] - sub[graph.lca_get_kth_ancestor(y, (dis1 + dis2) // 2 - 1)])
        return

    @staticmethod
    def cf_1328e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1328/E
        tag: tree_lca|dfs_order|classical
        """
        n, m = ac.read_list_ints()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        graph.lca_build_with_multiplication()
        for _ in range(m):
            nums = ac.read_list_ints_minus_one()[1:]
            deepest = nums[0]
            for num in nums:
                if graph.depth[num] > graph.depth[deepest]:
                    deepest = num
            for num in nums:
                fa = graph.lca_get_lca_between_nodes(num, deepest)
                if not (fa == num or graph.parent[num] == fa):
                    ac.st("NO")
                    break
            else:
                ac.st("YES")
        return

    @staticmethod
    def lc_1483():
        """
        url: https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/
        tag: sparse_table|tree_array|lca|tree_lca|classical
        """

        class TreeAncestor:

            def __init__(self, n: int, parent: List[int]):
                self.graph = WeightedTree(n)
                for i in range(1, n):
                    self.graph.add_directed_edge(parent[i], i, 1)
                self.graph.lca_build_with_multiplication()

            def getKthAncestor(self, node: int, k: int) -> int:
                return self.graph.lca_get_kth_ancestor(node, k)

        return TreeAncestor

    @staticmethod
    def lg_p3379_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3379
        tag: tree_lca|classical
        """
        n, m, s = ac.read_list_ints()
        s -= 1
        graph = WeightedTree(n, s)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        graph.lca_build_with_multiplication()
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            lca = graph.lca_get_lca_between_nodes(i, j) + 1
            ac.st(lca)
        return

    @staticmethod
    def lg_p3379_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3379
        tag: tree_lca|classical
        """
        n, m, r = ac.read_list_ints()
        r -= 1
        graph = HeavyChain(n, r)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j)
        graph.initial()
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ans = graph.query_lca(x, y) + 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p3379_3(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3379
        tag: tree_lca|classical
        """
        n, m, s = ac.read_list_ints()
        s -= 1
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)
        parent = [-1] * n
        stack = [s]
        parent[s] = s
        while stack:
            i = stack.pop()
            for j in edge[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        uf = UnionFindGetLCA(parent, s)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(uf.get_lca(x, y) + 1)
        return

    @staticmethod
    def lg_p3379_4(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3379
        tag: tree_lca|classical
        """
        n, m, s = ac.read_list_ints()
        s -= 1
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        queries = [ac.read_list_ints_minus_one() for _ in range(m)]
        ans = OfflineLCA().bfs_iteration(edge, queries, s)
        ac.st("\n".join(str(x + 1) for x in ans))
        return

    @staticmethod
    def cf_321c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/321/C
        tag: tree_centroid_recursion|classical
        """
        n = ac.read_int()
        to = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            to[u].append(v)
            to[v].append(u)

        cc, pp, ss = TreeCentroid().centroid_finder(to)
        ans = [64] * n
        for c, p in zip(cc, pp):
            ans[c] = ans[p] + 1
        ac.lst([chr(x) for x in ans])
        return

    @staticmethod
    def lg_p3384(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3384
        tag: tree_chain_split|tree_array|implemention
        """
        n, m, r, p = ac.read_list_ints()
        r -= 1
        tree = RangeAddRangeSum(n)
        nums = ac.read_list_ints()
        graph = HeavyChain(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j)
        graph.initial()
        tree.build([nums[i] for i in graph.rev_dfn])

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, z = lst[1:]
                path, _ = graph.query_chain(x - 1, y - 1)
                for a, b in path:
                    if a > b:
                        a, b = b, a
                    tree.range_add(a + 1, b + 1, z)
            elif lst[0] == 2:
                x, y = lst[1:]
                ans = 0
                path, _ = graph.query_chain(x - 1, y - 1)
                for a, b in path:
                    if a > b:
                        a, b = b, a
                    ans += tree.range_sum(a + 1, b + 1)
                    ans %= p
                ac.st(ans)
            elif lst[0] == 3:
                x, z = lst[1:]
                x -= 1
                a, b = graph.dfn[x], graph.cnt_son[x]
                tree.range_add(a + 1, a + b, z)
            else:
                x = lst[1] - 1
                a, b = graph.dfn[x], graph.cnt_son[x]
                ans = tree.range_sum(a + 1, a + b) % p
                ac.st(ans)
        return

    @staticmethod
    def lg_p2912(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2912
        tag: offline_lca|offline_query|classical
        """
        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append((j, w))
            dct[j].append((i, w))

        dis = [math.inf] * n
        dis[0] = 0
        parent = [-1] * n
        parent[0] = 0
        stack = [0]
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                if dis[j] == math.inf:
                    dis[j] = dis[i] + w
                    stack.append(j)
                    parent[j] = i
        tree = UnionFindGetLCA(parent)
        for x in range(q):
            i, j = ac.read_list_ints_minus_one()
            ans = dis[i] + dis[j] - 2 * dis[tree.get_lca(i, j)]
            ac.st(ans)
        return

    @staticmethod
    def lg_p3019(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3019
        tag: offline_query|lca
        """
        n, m = ac.read_list_ints()
        parent = [0] + [ac.read_int() - 1 for _ in range(n - 1)]
        tree = UnionFindGetLCA(parent)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(tree.get_lca(x, y) + 1)
        return

    @staticmethod
    def library_checker_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/lca
        tag: tree_lca
        """
        n, m = ac.read_list_ints()
        parent = [0] + ac.read_list_ints()
        uf = UnionFindGetLCA(parent, 0)
        ans = []
        for _ in range(m):
            x, y = ac.read_list_ints()
            ans.append(str(uf.get_lca(x, y)))
        ac.st("\n".join(ans))
        return

    @staticmethod
    def lc_2846(n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:

        """
        url: https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree
        tag: tree_lca|greed
        """

        class Graph(WeightedTree):
            def bfs_dis(self):
                stack = [0]
                while stack:
                    u = stack.pop()
                    for v, weight in self.get_to_nodes_weights(u):
                        if v != self.parent[u]:
                            dis[v] = dis[u][:]
                            dis[v][weight] += 1
                            stack.append(v)
                return

        graph = Graph(n)
        for i, j, w in edges:
            graph.add_undirected_edge(i, j, w - 1)
        graph.lca_build_with_multiplication()
        dis = [[0] * 26 for _ in range(n)]
        graph.bfs_dis()
        ans = []
        for i, j in queries:
            k = graph.lca_get_lca_between_nodes(i, j)
            cur = [dis[i][w] + dis[j][w] - 2 * dis[k][w] for w in range(26)]
            ceil = max(cur)
            tot = sum(cur)
            ans.append(tot - ceil)
        return ans

    @staticmethod
    def abc_294g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc294/tasks/abc294_g
        tag: segment_tree|point_set|range_sum|heavy_chain|tree_lca
        """
        n = ac.read_int()
        tree = PointSetRangeSum(n)
        edges = [ac.read_list_ints() for _ in range(n - 1)]
        graph = HeavyChain(n)
        for x in range(n - 1):
            i, j, _ = edges[x]
            graph.add_undirected_edge(i - 1, j - 1)
        graph.initial()
        val = [0] * n
        for i, j, w in edges:
            i -= 1
            j -= 1
            if graph.dfn[i] < graph.dfn[j]:
                val[j] = w
            else:
                val[i] = w

        nums = [val[i] for i in graph.rev_dfn]
        tree.build(nums)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, w = lst[1:]
                x -= 1
                i, j, _ = edges[x]
                i -= 1
                j -= 1
                if graph.dfn[i] < graph.dfn[j]:
                    tree.point_set(graph.dfn[j], w)
                    nums[graph.dfn[j]] = w
                else:
                    tree.point_set(graph.dfn[i], w)
                    nums[graph.dfn[i]] = w

            else:
                x, y = lst[1:]
                y -= 1
                x -= 1
                ans = 0
                lst, lca = graph.query_chain(x, y)
                for a, b in lst:
                    if a <= b:
                        ans += tree.range_sum(a, b)
                    else:
                        ans += tree.range_sum(b, a)
                ans -= nums[lca]
                ac.st(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/BITTREEMIN
        tag: data_range|minimum_xor|tree_lca
        """
        cnt = [0] * 1001
        for _ in range(ac.read_int()):
            n, q = ac.read_list_ints()
            graph = WeightedTree(n)
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)
            graph.lca_build_with_multiplication()
            nums = [x // 2 for x in ac.read_list_ints()]
            for _ in range(q):
                op, u, v = ac.read_list_ints_minus_one()
                if op == 0:
                    nums[u] = (v + 1) // 2
                else:
                    dis = graph.lca_get_lca_and_dist_between_nodes(u, v)[1]
                    if dis > 1001:
                        ac.st(0)
                    else:
                        lca = graph.lca_get_lca_between_nodes(u, v)
                        lst = [u]
                        while lst[-1] != lca:
                            lst.append(graph.parent[lst[-1]])
                        while v != lca:
                            lst.append(v)
                            v = graph.parent[v]
                        for i in range(1001):
                            cnt[i] = 0
                        for x in lst:
                            cnt[nums[x]] += 1
                        ans = pre = math.inf
                        for i in range(1001):
                            if cnt[i] >= 2:
                                ans = 0
                                break
                            if cnt[i]:
                                if pre < math.inf:
                                    ans = min(ans, pre ^ i)
                                pre = i
                        ac.st(ans)
        return

    @staticmethod
    def abc_209d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc209/tasks/abc209_d
        tag: tree_ancestor
        """
        n, q = ac.read_list_ints()
        tree = WeightedTree(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            tree.add_undirected_edge(i, j)
        tree.lca_build_with_multiplication()
        for _ in range(q):
            i, j = ac.read_list_ints_minus_one()
            dis = tree.lca_get_lca_and_dist_between_nodes(i, j)[1]
            if dis % 2:
                ac.st("Road")
            else:
                ac.st("Town")
        return

    @staticmethod
    def cf_343d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/343/problem/D
        tag: heavy_chain|range_set|point_get|classical|implemention
        """
        n = ac.read_int()
        graph = HeavyChain(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j)
        graph.initial()
        tree = RangeSetPointGet(n, -1)
        tree.build([0] * n)
        for _ in range(ac.read_int()):
            op, v = ac.read_list_ints_minus_one()
            if op == 0:
                start = graph.dfn[v]
                end = start + graph.cnt_son[v] - 1
                tree.range_set(start, end, 1)
            elif op == 1:
                x, y = v, 0
                path, _ = graph.query_chain(x, y)
                for a, b in path:
                    if a > b:
                        a, b = b, a
                    tree.range_set(a, b, 0)
            else:
                ans = tree.point_get(graph.dfn[v])  # important!!!
                ac.st(ans)
        return

    @staticmethod
    def cf_1843f2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1843/problem/F2
        tag: tree_lca|multiplication_method|classical|max_con_sub_sum
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            query = []
            dct = [[]]
            sub = [1]
            for _ in range(n):
                lst = ac.read_list_strs()
                if lst[0] == "+":
                    v, x = [int(w) for w in lst[1:]]
                    v -= 1
                    sub.append(x)
                    dct[v].append(len(sub) - 1)
                    dct.append([])
                else:
                    u, v, k = [int(w) - 1 for w in lst[1:]]
                    query.append((u, v, k + 1))

            ceil = TreeAncestorMaxSub(dct, sub[:])
            floor = TreeAncestorMaxSub(dct, [-x for x in sub])
            for u, v, k in query:
                high = ceil.get_max_con_sum(u, v)
                low = -floor.get_max_con_sum(u, v)
                ac.st("YES" if low <= k <= high or k == 0 else "NO")
        return

    @staticmethod
    def cf_1304e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1304/E
        tag: observation|tree_lca|graph|implemention
        """
        n = ac.read_int()
        tree = WeightedTree(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            tree.add_undirected_edge(i, j)
        tree.lca_build_with_multiplication()
        for _ in range(ac.read_int()):
            x, y, a, b, k = ac.read_list_ints_minus_one()
            k += 1
            dis = tree.lca_get_lca_and_dist_between_nodes(a, x)[1] + 1
            dis += tree.lca_get_lca_and_dist_between_nodes(y, b)[1]
            if dis <= k and (k - dis) % 2 == 0:
                ac.yes()
                continue
            dis = tree.lca_get_lca_and_dist_between_nodes(a, y)[1] + 1
            dis += tree.lca_get_lca_and_dist_between_nodes(x, b)[1]
            if dis <= k and (k - dis) % 2 == 0:
                ac.yes()
                continue
            dis = tree.lca_get_lca_and_dist_between_nodes(a, b)[1]
            if dis <= k and (k - dis) % 2 == 0:
                ac.yes()
                continue
            ac.no()
        return

    @staticmethod
    def lg_p3976(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3976
        tag: range_add|range_max_gain|range_min_gain
        """
        inf = 10 ** 18
        n = ac.read_int()
        nums = ac.read_list_ints()
        graph = HeavyChain(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j)
        graph.initial()
        nums = [nums[i] for i in graph.rev_dfn]
        tree = RangeAddRangeMaxGainMinGain(n, inf)
        tree.build(nums)

        for _ in range(ac.read_int()):
            a, b, v = ac.read_list_ints()
            a -= 1
            b -= 1
            ans = [inf, -inf, -inf, inf]  # floor, ceil, max_gain, min_gain
            path, _ = graph.query_chain(a, b)
            for x, y in path:
                if x <= y:
                    floor, ceil, max_gain, min_gain = tree.range_max_gain_min_gain(x, y)
                else:
                    floor, ceil, aa, bb = tree.range_max_gain_min_gain(y, x)
                    max_gain = -bb
                    min_gain = -aa
                ans[2] = max(ans[2], max_gain, ceil - ans[0])
                ans[3] = min(ans[3], min_gain, floor - ans[1])
                ans[0] = min(ans[0], floor)
                ans[1] = max(ans[1], ceil)
            for x, y in path:
                if x <= y:
                    tree.range_add(x, y, v)
                else:
                    tree.range_add(y, x, v)
            ac.st(max(ans[2], 0))
        return

    @staticmethod
    def abc_202e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc202/tasks/abc202_e
        tag: heuristic_merge|offline_query|classical
        """
        n = ac.read_int()
        p = [-1] + ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        for i in range(1, n):
            dct[p[i]].append(i)

        q = ac.read_int()
        ans = [-1] * q
        sub = [dict() for _ in range(n)]
        node = list(range(n))
        queries = [[] for _ in range(n)]
        for i in range(q):
            u, d = ac.read_list_ints()
            u -= 1
            queries[u].append((d, i))
        depth = [0] * n
        stack = [0]
        while stack:
            x = stack.pop()
            if x >= 0:
                stack.append(~x)
                for y in dct[x]:
                    depth[y] = depth[x] + 1
                    stack.append(y)
            else:
                x = ~x
                cur = node[x]
                sub[cur][depth[x]] = 1
                for y in dct[x]:
                    nex = node[y]
                    if len(sub[cur]) < len(sub[nex]):
                        cur, nex = nex, cur
                    for kk, vv in sub[nex].items():
                        sub[cur][kk] = sub[cur].get(kk, 0) + vv
                node[x] = cur
                for dd, ii in queries[x]:
                    ans[ii] = sub[cur].get(dd, 0)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def abc_187e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc187/tasks/abc187_e
        tag: tree_diff_array|tree_dfs_order|tree_lca
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        edges = []
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
            edges.append((i, j))
        graph.dfs_order()
        graph.lca_build_with_multiplication()
        diff = [0] * (n + 1)
        for _ in range(ac.read_int()):
            t, e, x = ac.read_list_ints()
            e -= 1
            i, j = edges[e]
            if t == 2:
                i, j = j, i
            lca, dis = graph.lca_get_lca_and_dist_between_nodes(i, j)
            if lca == j:
                j = graph.lca_get_kth_ancestor(i, dis - 1)
                a, b = graph.start[j], graph.end[j]
                diff[a] += x
                diff[b + 1] -= x
            else:
                diff[0] += x
                diff[graph.start[j]] -= x
                diff[graph.end[j] + 1] += x
        ans = [0] * n
        for i in range(1, n):
            diff[i] += diff[i - 1]
        for i in range(n):
            ans[i] = diff[graph.start[i]]
        ac.flatten(ans)
        return
