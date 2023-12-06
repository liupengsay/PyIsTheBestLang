"""

Algorithm：LCA、倍增算法、树链剖分、树的质心、树的重心、离线LCA与树上差分
Function：来求一棵树的最近公共祖先（LCA）也可以使用

====================================LeetCode====================================

=====================================LuoGu======================================

==================================LibreOJ==================================

===================================CodeForces===================================

====================================AtCoder=====================================

=====================================AcWing=====================================



"""
import math

from src.data_structure.tree_array.template import RangeAddRangeSum
from src.graph.tree_lca.template import OfflineLCA, TreeAncestor, TreeCentroid, HeavyChain, TreeAncestorPool
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p7167(ac=FastIO()):

        # 模板：单调栈加倍增LCA计算
        n, q = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        # 使用单调栈建树
        parent = [n] * n
        edge = [[] for _ in range(n + 1)]
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]][0] < nums[i][0]:
                parent[stack.pop()] = i
            stack.append(i)
        for i in range(n):
            edge[n - parent[i]].append(n - i)

        # LCA预处理
        weight = [x for _, x in nums] + [math.inf]
        tree = TreeAncestorPool(edge, weight[::-1])

        # 查询
        for _ in range(q):
            r, v = ac.read_list_ints()
            ans = tree.get_final_ancestor(n - r + 1, v)
            ac.st(0 if ans == 0 else n - ans + 1)
        return

    @staticmethod
    def cf_519e(ac=FastIO()):
        # 模板：使用LCA计算第k个祖先与节点之间的距离
        n = ac.read_int()
        edges = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edges[x].append(y)
            edges[y].append(x)

        lca = TreeAncestor(edges)
        sub = [0] * n

        @ac.bootstrap
        def dfs(i, fa):
            nonlocal sub
            cur = 1
            for j in edges[i]:
                if j != fa:
                    yield dfs(j, i)
                    cur += sub[j]
            sub[i] = cur
            yield

        dfs(0, -1)

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
        # 模板：利用 LCA 的方式查询是否为一条链上距离不超过 1 的点
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
        # 模板：查询任意节点的第 k 个祖先
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
        # 模板：使用倍增查询任意两个节点的 LCA
        n, m, s = ac.read_list_ints()
        s -= 1
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)
        # 需要改 s 为默认根
        tree = TreeAncestor(edge)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(tree.get_lca(x, y) + 1)
        return

    @staticmethod
    def abc_70d(ac=FastIO()):
        # 模板：典型LCA查询运用题，也可离线实现
        n = ac.read_int()
        edges = [[] for _ in range(n)]
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            a, b, c = ac.read_list_ints()
            dct[a - 1].append([b - 1, c])
            dct[b - 1].append([a - 1, c])
            edges[a - 1].append(b - 1)
            edges[b - 1].append(a - 1)
        tree = TreeAncestor(edges)

        q, k = ac.read_list_ints()
        k -= 1
        dis = [0] * n
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j, w in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    dis[j] = dis[i] + w
        for _ in range(q):
            a, b = ac.read_list_ints_minus_one()
            c1 = tree.get_lca(a, k)
            c2 = tree.get_lca(b, k)
            ans1 = dis[a] + dis[k] - 2 * dis[c1]
            ans2 = dis[b] + dis[k] - 2 * dis[c2]
            ac.st(ans1 + ans2)
        return

    @staticmethod
    def cf_321c(ac=FastIO()):
        # 模板：使用质心算法进行树的递归切割
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
        # 模板：使用树链剖分和深搜序进行节点值修改与区间和查询
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
                    tree.update_range(a + 1, b + 1, z)
            elif lst[0] == 2:
                x, y = lst[1:]
                ans = 0
                for a, b in heavy.query_chain(x - 1, y - 1):
                    ans += tree.get_sum_range(a + 1, b + 1)
                    ans %= p
                ac.st(ans)
            elif lst[0] == 3:
                x, z = lst[1:]
                x -= 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
                tree.update_range(a + 1, a + b, z)
            else:
                x = lst[1] - 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
                ans = tree.get_sum_range(a + 1, a + b) % p
                ac.st(ans)
        return

    @staticmethod
    def lg_p3379_2(ac=FastIO()):
        # 模板：使用树链剖分求 LCA
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
    def lg_p2912(ac=FastIO()):
        # 模板：离线LCA查询与任意点对之间距离计算
        n, q = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i][j] = dct[j][i] = w

        # 根节点bfs计算距离
        dis = [math.inf] * n
        dis[0] = 0
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    dis[j] = dis[i] + dct[i][j]
                    stack.append([j, i])

        # 查询公共祖先
        queries = [ac.read_list_ints_minus_one() for _ in range(q)]
        dct = [list(d.keys()) for d in dct]
        ancestor = OfflineLCA().bfs_iteration(dct, queries, 0)
        for x in range(q):
            i, j = queries[x]
            w = ancestor[x]
            ans = dis[i] + dis[j] - 2 * dis[w]
            ac.st(ans)
        return

    @staticmethod
    def lg_p3019(ac=FastIO()):
        # 模板：离线查询 LCA 最近公共祖先
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for i in range(n - 1):
            dct[ac.read_int() - 1].append(i + 1)
        queries = [ac.read_list_ints_minus_one() for _ in range(m)]
        ans = OfflineLCA().bfs_iteration(dct, queries)
        for a in ans:
            ac.st(a + 1)
        return

    @staticmethod
    def lg_p3258(ac=FastIO()):
        # 模板：离线LCA加树上差分加树形DP
        n = ac.read_int()
        nums = ac.read_list_ints_minus_one()
        root = nums[0]
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        trips = []
        for i in range(1, n):
            trips.append([nums[i - 1], nums[i]])

        # 离线LCA
        res = OfflineLCA().bfs_iteration(dct, trips, root)
        # 树上差分
        diff = [0] * n
        for i in range(n - 1):
            u, v, ancestor = trips[i] + [res[i]]
            # 将 u 与 v 到 ancestor 的路径经过的节点进行差分修改（不包含u）
            if u != ancestor:
                u = parent[u]
                diff[u] += 1
                diff[v] += 1
                diff[ancestor] -= 1
                if parent[ancestor] != -1:
                    diff[parent[ancestor]] -= 1
            else:
                diff[v] += 1
                diff[u] -= 1

        # 自底向上进行差分加和
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append(j)
            else:
                i = ~i
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        diff[nums[0]] += 1
        diff[nums[-1]] -= 1
        for a in diff:
            ac.st(a)
        return

    @staticmethod
    def lg_p6969(ac=FastIO()):
        # 模板：离线 LCA 查询与树上边差分计算
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        cost = [dict() for _ in range(n)]
        for _ in range(n - 1):
            a, b, c1, c2 = ac.read_list_ints()
            a -= 1
            b -= 1
            cost[a][b] = cost[b][a] = [c1, c2]
            dct[a].append(b)
            dct[b].append(a)

        query = [[i, i + 1] for i in range(n - 1)]
        res = OfflineLCA().bfs_iteration(dct, query)
        for i in range(n - 1):
            query[i].append(res[i])
        diff = TreeDiffArray().bfs_iteration_edge(dct, query, 0)

        # 将变形的边差分还原
        ans = 0
        stack = [0]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i
                    # 边计数下放到节点上
                    cnt = diff[j]
                    c1, c2 = cost[i][j]
                    ans += ac.min(cnt * c1, c2)
        ac.st(ans)
        return