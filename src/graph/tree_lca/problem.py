"""

Algorithm：lca|multiplication_method|tree_chain_split|offline_lca|online_lca
Description：

====================================LeetCode====================================
1483（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）sparse_table|tree_array|lca|tree_lca|classical
2846（https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree）tree_lca|greedy

=====================================LuoGu======================================
P3379（https://www.luogu.com.cn/problem/P3379）tree_lca|classical
P7128（https://www.luogu.com.cn/problem/P7128）lca|implemention|sort
P7167（https://www.luogu.com.cn/problem/P7167）monotonic_stack|tree_lca|build_tree
P2912（https://www.luogu.com.cn/problem/P2912）offline_lca|offline_query
P3019（https://www.luogu.com.cn/problem/P3019）offline_query|lca
P3384（https://www.luogu.com.cn/problem/P3384）tree_chain_split|tree_array|implemention

======================================LibraryChecker==================================
1（https://judge.yosupo.jp/problem/lca）tree_lca

===================================CodeForces===================================
1328E（https://codeforces.com/problemset/problem/1328/E）tree_lca|dfs_order
321C（https://codeforces.com/problemset/problem/321/C）tree_centroid_recursion|classical
519E（https://codeforces.com/problemset/problem/519/E）lca|kth_ancestor|counter
1296F（https://codeforces.com/contest/1296/problem/F）offline_lca|greedy|construction|multiplication_method
1702G2（https://codeforces.com/contest/1702/problem/G2）tree_lca

====================================AtCoder=====================================
ABC294G（https://atcoder.jp/contests/abc294/tasks/abc294_g）segment_tree|point_set|range_sum|heavy_chain|tree_lca
ABC209D（https://atcoder.jp/contests/abc209/tasks/abc209_d）tree_ancestor

=====================================AcWing=====================================
4202（https://www.acwing.com/problem/content/4205/）bit_operation|build_graph|tree_lca|tree_dis

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/BITTREEMIN）data_range|minimum_xor|tree_lca


"""
import math
from typing import List

from src.data_structure.segment_tree.template import PointSetRangeSum
from src.data_structure.tree_array.template import RangeAddRangeSum
from src.graph.tree_diff_array.template import TreeDiffArray
from src.graph.tree_lca.template import OfflineLCA, TreeAncestor, TreeCentroid, HeavyChain, TreeAncestorPool, \
    UnionFindGetLCA
from src.utils.fast_io import FastIO, inf


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

        weight = [x for _, x in nums] + [inf]
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
        n = ac.read_int()
        edges = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edges[x].append(y)
            edges[y].append(x)

        lca = TreeAncestor(edges)
        sub = [0] * n
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in edges[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                cur = 1
                for j in edges[i]:
                    if j != fa:
                        cur += sub[j]
                sub[i] = cur

        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            if x == y:
                ac.st(n)
                continue

            dis = lca.get_dist(x, y)
            if dis % 2 == 1:
                ac.st(0)
            else:
                z = lca.get_lca(x, y)
                dis1 = lca.get_dist(x, z)
                dis2 = lca.get_dist(y, z)
                if dis1 == dis2:
                    up = n - sub[z]
                    down = sub[z] - sub[lca.get_kth_ancestor(x, dis1 - 1)] - sub[lca.get_kth_ancestor(y, dis2 - 1)]
                    ac.st(up + down)
                elif dis1 > dis2:
                    w = lca.get_kth_ancestor(x, (dis1 + dis2) // 2)
                    ac.st(sub[w] - sub[lca.get_kth_ancestor(x, (dis1 + dis2) // 2 - 1)])
                else:
                    w = lca.get_kth_ancestor(y, (dis1 + dis2) // 2)
                    ac.st(sub[w] - sub[lca.get_kth_ancestor(y, (dis1 + dis2) // 2 - 1)])
        return

    @staticmethod
    def cf_1328e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1328/E
        tag: tree_lca|dfs_order|classical
        """
        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        tree = TreeAncestor(edge)
        for _ in range(m):
            nums = ac.read_list_ints_minus_one()[1:]
            deep = nums[0]
            for num in nums:
                if tree.depth[num] > tree.depth[deep]:
                    deep = num
            ans = True
            for num in nums:
                fa = tree.get_lca(num, deep)
                if fa == num or tree.parent[num] == fa:
                    continue
                else:
                    ans = False
                    break
            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def lc_1483(parent, node, k):
        """
        url: https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/
        tag: sparse_table|tree_array|lca|tree_lca|classical
        """
        n = len(parent)
        edges = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                edges[i].append(parent[i])
                edges[parent[i]].append(i)
        tree = TreeAncestor(edges)
        return tree.get_kth_ancestor(node, k)

    @staticmethod
    def lg_p3379_1(ac=FastIO()):
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

        tree = TreeAncestor(edge, s)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(tree.get_lca(x, y) + 1)
        return

    @staticmethod
    def lg_p3379_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3379
        tag: tree_lca|classical
        """
        n, m, r = ac.read_list_ints()
        r -= 1
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        heavy = HeavyChain(dct, r)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(heavy.query_lca(x, y) + 1)
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
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        heavy = HeavyChain(dct, r)
        tree.build([nums[i] for i in heavy.rev_dfn])

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, z = lst[1:]
                for a, b in heavy.query_chain(x - 1, y - 1):
                    tree.range_add(a + 1, b + 1, z)
            elif lst[0] == 2:
                x, y = lst[1:]
                ans = 0
                for a, b in heavy.query_chain(x - 1, y - 1):
                    ans += tree.range_sum(a + 1, b + 1)
                    ans %= p
                ac.st(ans)
            elif lst[0] == 3:
                x, z = lst[1:]
                x -= 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
                tree.range_add(a + 1, a + b, z)
            else:
                x = lst[1] - 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
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

        dis = [inf] * n
        dis[0] = 0
        parent = [-1] * n
        parent[0] = 0
        stack = [0]
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                if dis[j] == inf:
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
        tag: tree_lca|greedy
        """
        dct = [[] for _ in range(n)]
        for i, j, _ in edges:
            dct[i].append(j)
            dct[j].append(i)
        tree = TreeAncestor(dct)

        dct = [[] for _ in range(n)]
        for i, j, w in edges:
            dct[i].append([j, w - 1])
            dct[j].append([i, w - 1])
        cnt = [[0] * 26 for _ in range(n)]
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            for j, w in dct[i]:
                if j != fa:
                    cnt[j] = cnt[i][:]
                    cnt[j][w] += 1
                    stack.append((j, i))
        ans = []
        for i, j in queries:
            k = tree.get_lca(i, j)
            cur = [cnt[i][w] + cnt[j][w] - 2 * cnt[k][w] for w in range(26)]
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
        dct = [[] for _ in range(n)]
        for x in range(n - 1):
            i, j, w = edges[x]
            i -= 1
            j -= 1
            dct[i].append(j)
            dct[j].append(i)
        heavy = HeavyChain(dct, 0)

        val = [0] * n
        for i, j, w in edges:
            i -= 1
            j -= 1
            if heavy.dfn[i] < heavy.dfn[j]:
                val[j] = w
            else:
                val[i] = w

        tree.build([val[i] for i in heavy.rev_dfn])

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, w = lst[1:]
                x -= 1
                i, j, _ = edges[x]
                i -= 1
                j -= 1
                if heavy.dfn[i] < heavy.dfn[j]:
                    tree.point_set(heavy.dfn[j], w)
                else:
                    tree.point_set(heavy.dfn[i], w)

            else:
                x, y = lst[1:]
                y -= 1
                x -= 1
                if heavy.dfn[y] < heavy.dfn[x]:
                    x, y = y, x
                ans = 0
                lca = heavy.dfn[heavy.query_lca(x, y)]
                for a, b in heavy.query_chain(x, y):
                    if a <= lca <= b:
                        lst = [(a, lca - 1), (lca + 1, b)]
                    else:
                        lst = [(a, b)]
                    for aa, bb in lst:
                        if 0 <= aa <= bb < n:
                            cur = tree.range_sum(aa, bb)
                            ans += cur
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
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)

            nums = [x//2 for x in ac.read_list_ints()]

            tree = TreeAncestor(dct, 0)
            for _ in range(q):
                op, u, v = ac.read_list_ints_minus_one()
                if op == 0:
                    nums[u] = (v + 1) // 2
                else:
                    dis = tree.get_dist(u, v)
                    if dis > 1001:
                        ac.st(0)
                    else:
                        lca = tree.get_lca(u, v)
                        lst = [u]
                        while lst[-1] != lca:
                            lst.append(tree.parent[lst[-1]])
                        while v != lca:
                            lst.append(v)
                            v = tree.parent[v]
                        for i in range(1001):
                            cnt[i] = 0
                        for x in lst:
                            cnt[nums[x]] += 1
                        ans = pre = inf
                        for i in range(1001):
                            if cnt[i] >= 2:
                                ans = 0
                                break
                            if cnt[i]:
                                if pre < inf:
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
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        tree = TreeAncestor(dct, 0)
        for _ in range(q):
            i, j = ac.read_list_ints_minus_one()
            dis = tree.get_dist(i, j)
            if dis % 2:
                ac.st("Road")
            else:
                ac.st("Town")
        return