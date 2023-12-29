"""

Algorithm：mst|kruskal|prim|strictly_second_mst|multiplication_method|lca|brute_force|shortest_path_spanning_tree|greedy
Description：prim is node wise and kruskal is edge wise, prim is suitable for dense graph and kruskal is suitable for sparse graph


====================================LeetCode====================================
1489（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）mst|necessary_edge|fake_necessary_edge
1584（https://leetcode.cn/problems/min-cost-to-connect-all-points/）dense_graph|prim|mst
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
P6705（https://www.luogu.com.cn/problem/P6705）brute_force|mst
P7775（https://www.luogu.com.cn/problem/P7775）bfs|mst_like
P2658（https://www.luogu.com.cn/problem/P2658）classical|mst
P4180（https://www.luogu.com.cn/problem/P4180）mst|lca|multiplication_method|strictly_second_mst|classical
P1265（https://www.luogu.com.cn/problem/P1265）prim|mst
P1340（https://www.luogu.com.cn/problem/P1340）reverse_order|union_find|mst
P1550（https://www.luogu.com.cn/problem/P1550）build_graph|mst|fake_source
P2212（https://www.luogu.com.cn/problem/P2212）prim|mst|dense_graph
P2847（https://www.luogu.com.cn/problem/P2847）prim|mst|dense_graph
P3535（https://www.luogu.com.cn/problem/P3535）mst|judge_circle_by_union_find|connected
P4047（https://www.luogu.com.cn/problem/P4047）mst
P6171（https://www.luogu.com.cn/problem/P6171）sparse|kruskal|mst
P1550（https://www.luogu.com.cn/problem/P1550）mst|build_graph|fake_source|classical

===================================CodeForces===================================
472D（https://codeforces.com/problemset/problem/472/D）mst|construction|shortest_path|prim
609E（https://codeforces.com/problemset/problem/609/E）lca|greedy|mst|strictly_second_mst|necessary_edge
1108F（https://codeforces.com/contest/1108/problem/F）mst

====================================AtCoder=====================================
ARC076B（https://atcoder.jp/contests/abc065/tasks/arc076_b）mst

=====================================AcWing=====================================
3728（https://www.acwing.com/problem/content/3731/）prim|mst|dense_graph|specific_plan

================================LibraryChecker================================
Manhattan MST（https://judge.yosupo.jp/problem/manhattanmst）
Directed MST（https://judge.yosupo.jp/problem/directedmst）

"""
import math
from collections import defaultdict
from heapq import heappop, heappush
from typing import List

from src.graph.minimum_spanning_tree.template import TreeAncestorWeightSecond, MinimumSpanningTree
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1991(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1991
        tag: mst|connected_graph
        """

        k, n = ac.read_list_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            for j in range(i + 1, n):
                a = pos[i][0] - pos[j][0]
                b = pos[i][1] - pos[j][1]
                edge.append([i, j, a * a + b * b])

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
    def lg_2820(ac=FastIO()):
        # 求删除最大权值和使得存在回路的连通图变成mst|
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
        # kruskal求mst|
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = MinimumSpanningTree(edges, n, "kruskal")
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
        # prim求mst|
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = MinimumSpanningTree(edges, n, "prim")
        if mst.cnt < n:
            ac.st("orz")
        else:
            ac.st(mst.cost)
        return

    @staticmethod
    def cf_1108f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1108/problem/F
        tag: mst
        """
        # 使得mst|的边组合唯一时，需要增|权重的最少边数量
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            if i != j:  # 去除自环
                edges.append([i - 1, j - 1, w])

        # kruskalmst|
        uf = UnionFind(n)
        dct = [dict() for _ in range(n)]
        cost = 0
        for i, j, w in sorted(edges, key=lambda it: it[2]):
            if uf.union(i, j):
                cost += w
                dct[i][j] = dct[j][i] = w
            if uf.part == 1:
                break
        del uf
        # brute_force新增的边
        tree = TreeAncestorWeightSecond(dct)
        ans = 0
        # 使得mst|唯一等价于有某条边参与时依旧代价最小的该边数量
        for i, j, w in edges:
            if j in dct[i] and dct[i][j] == w:
                ans += 1
            else:
                dis = tree.get_dist_weight_max_second(i, j)[0]
                if dis == w:
                    ans += 1
        ac.st(ans - n + 1)
        return

    @staticmethod
    def lc_1489(n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
        tag: mst|necessary_edge|fake_necessary_edge
        """
        # 求mst|的关键边与伪关键边
        m = len(edges)
        # 代价sorting
        lst = list(range(m))
        lst.sort(key=lambda it: edges[it][2])

        # mst|代价
        min_cost = 0
        uf = UnionFind(n)
        for i in lst:
            x, y, cost = edges[i]
            if uf.union(x, y):
                min_cost += cost

        # brute_force关键边
        key = set()
        for i in lst:
            cur_cost = 0
            uf = UnionFind(n)
            for j in lst:
                if j != i:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
            if cur_cost > min_cost or uf.part != 1:
                key.add(i)

        # brute_force伪关键边
        fake = set()
        for i in lst:
            if i not in key:
                cur_cost = edges[i][2]
                uf = UnionFind(n)
                # 先将当前边|入生成树
                uf.union(edges[i][0], edges[i][1])
                for j in lst:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
                # 若仍然是mst|就说明是伪关键边
                if cur_cost == min_cost and uf.part == 1:
                    fake.add(i)

        return [list(key), list(fake)]

    @staticmethod
    def lg_p2872(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2872
        tag: prim|mst
        """

        # primmst|，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res ** 0.5

        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for i in range(m):
            u, v = ac.read_list_ints_minus_one()
            dct[u][v] = dct[v][u] = 0
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            nex = -1
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                visit[j] = ac.min(visit[j], dct[i].get(j, inf))
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1194(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1194
        tag: mst
        """
        # 超级源点build_graph|mst|
        a, b = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(b)]
        edge = [[0, i, a] for i in range(1, b + 1)]
        for i in range(b):
            for j in range(i + 1, b):
                if 0 < grid[i][j] < a:
                    edge.append([i + 1, j + 1, grid[i][j]])
        mst = MinimumSpanningTree(edge, b + 1)
        ac.st(mst.cost)
        return

    @staticmethod
    def cf_472d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/472/D
        tag: mst|construction|shortest_path|prim
        """
        #  prim 校验mst|是否存在
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            if grid[i][i]:
                ac.st("NO")
                return
            for j in range(i + 1, n):
                if grid[i][j] != grid[j][i] or not grid[i][j]:
                    ac.st("NO")
                    return

        # Prim greedy按照权值选择边连通合并
        dis = [math.inf] * n
        dis[0] = 0
        visit = [0] * n
        stack = [[0, 0, -1]]
        res = [[] for _ in range(n)]
        cnt = 0
        while stack:
            d, i, fa = heappop(stack)
            if visit[i]:
                continue
            visit[i] = 1
            if fa != -1:
                res[fa].append([i, d])
                res[i].append([fa, d])
            cnt += 1
            if cnt == n:
                break
            for j in range(n):
                w = grid[i][j]
                if w < dis[j]:
                    dis[j] = w
                    heappush(stack, [w, j, i])
        del stack

        # bfs 根节点到所有节点的距离
        for i in range(n):
            cur = [inf] * n
            stack = [i]
            cur[i] = 0
            while stack:
                x = stack.pop()
                for y, w in res[x]:
                    if cur[y] == inf:
                        cur[y] = cur[x] + w
                        stack.append(y)
            if cur != grid[i]:
                ac.st("NO")
                return
        ac.st("YES")
        return

    @staticmethod
    def lg_p4180(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4180
        tag: mst|lca|multiplication_method|strictly_second_mst|classical
        """
        # mst|与LCA倍增查询strictly_second_mst
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            if i != j:  # 去除自环
                edges.append([i - 1, j - 1, w])

        # kruskalmst|
        edges.sort(key=lambda it: it[2])
        uf = UnionFind(n)
        dct = [dict() for _ in range(n)]
        cost = 0
        for i, j, w in edges:
            if uf.union(i, j):
                cost += w
                dct[i][j] = dct[j][i] = w
            if uf.part == 1:
                break

        # brute_force新增的边
        tree = TreeAncestorWeightSecond(dct)
        ans = inf
        for i, j, w in edges:
            for dis in tree.get_dist_weight_max_second(i, j):
                if dis != -1:
                    cur = cost - dis + w
                    if cost < cur < ans:
                        ans = cur
        ac.st(ans)
        return

    @staticmethod
    def cf_609e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/609/E
        tag: lca|greedy|mst|strictly_second_mst|necessary_edge
        """
        # mst|有指定边参与时的最小权值和，由此也可strictly_second_mst
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            if i != j:  # 去除自环
                edges.append([i - 1, j - 1, w])

        # kruskalmst|
        uf = UnionFind(n)
        dct = [dict() for _ in range(n)]
        cost = 0
        for i, j, w in sorted(edges, key=lambda it: it[2]):
            if uf.union(i, j):
                cost += w
                dct[i][j] = dct[j][i] = w
            if uf.part == 1:
                break

        # brute_force新增的边
        tree = TreeAncestorWeightSecond(dct)
        for i, j, w in edges:
            if j in dct[i] and dct[i][j] == w:
                ac.st(cost)
            else:
                dis = tree.get_dist_weight_max_second(i, j)[0]
                ac.st(cost - dis + w)
        return

    @staticmethod
    def lg_p1265(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1265
        tag: prim|mst
        """

        # primmst|，适合稠密图场景

        def dis(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        # 初始化最短距离
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d ** 0.5
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1340(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1340
        tag: reverse_order|union_find|mst
        """
        # reverse_order|union_find，维护mst|的边
        n, w = ac.read_list_ints()

        # offline_query处理，按照边权sorting
        edges = [ac.read_list_ints() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: edges[it][-1])

        uf = UnionFind(n)
        ans = []
        select = set()
        cost = 0
        for i in range(w - 1, -1, -1):
            if uf.part > 1:
                # 重新生成mst|
                cost = 0
                select = set()
                for j in ind:
                    if j <= i:
                        x, y, ww = edges[j]
                        if uf.union(x - 1, y - 1):
                            cost += ww
                            select.add(j)
            if uf.part > 1:
                # 无法连通直接终止
                ans.append(-1)
                break
            ans.append(cost)
            if i in select:  # 当前路径不可用，重置union_find
                uf = UnionFind(n)
                select = set()
                cost = 0
        while len(ans) < w:
            ans.append(-1)

        for i in range(w - 1, -1, -1):
            ls = ans[i]
            ac.st(ls)
        return

    @staticmethod
    def lg_p1550(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1550
        tag: mst|build_graph|fake_source|classical
        """
        # 建立虚拟源点，转换为mst|问题
        n = ac.read_int()
        edges = []
        for i in range(n):
            w = ac.read_int()
            edges.append([0, i + 1, w])

        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i + 1, n):
                edges.append([i + 1, j + 1, grid[j]])
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

        # primmst|，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res if res >= c else inf

        n, c = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p2658(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2658
        tag: classical|mst
        """
        # classicalmst|
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        uf = UnionFind(m * n)
        s = 0
        for i in range(m):
            lst = ac.read_list_ints()
            for j in range(n):
                if lst[j]:
                    uf.size[i * n + j] = 1
                    s += 1
        edge = []
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    edge.append([i * n + j, i * n + j + n, abs(grid[i][j] - grid[i + 1][j])])
                if j + 1 < n:
                    edge.append([i * n + j, i * n + j + 1, abs(grid[i][j] - grid[i][j + 1])])
        edge.sort(key=lambda it: it[2])
        del grid

        if s == 1:
            ac.st(0)
            return
        for x, y, d in edge:
            uf.union(x, y)
            if uf.size[uf.find(x)] == s:
                ac.st(d)
                return
        return

    @staticmethod
    def lg_p2847(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2847
        tag: prim|mst|dense_graph
        """

        # primmst|，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res

        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans = ac.max(ans, d)
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p3535(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3535
        tag: mst|judge_circle_by_union_find|connected
        """
        # mst|思想与union_find判环
        n, m, k = ac.read_list_ints()
        edge = []
        uf = UnionFind(n)
        ans = 0
        for _ in range(m):
            # 先将大于等于 k 的连接起来
            i, j = ac.read_list_ints_minus_one()
            if i >= k and j >= k:
                uf.union(i, j)
            else:
                edge.append([i, j])
        # 再依次判断剩余的边
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

        # mst|最优聚类距离
        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            x1, y1 = nums[i]
            for j in range(i + 1, n):
                x2, y2 = nums[j]
                edge.append([i, j, dis()])
        edge.sort(key=lambda it: it[2])

        # 当分成 k 个联通块时最短距离
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
        # 稀疏图 Kruskal mst|
        a, b, n, m = ac.read_list_ints()
        nums1 = [0, a] + [ac.read_int() for _ in range(n)]
        nums2 = [0, b] + [ac.read_int() for _ in range(m)]
        nums1.sort()
        nums2.sort()
        dct = defaultdict(list)
        for i in range(1, m + 2):
            for j in range(1, n + 2):
                # build_graph|是关键
                x = (i - 1) * (n + 1) + (j - 1)
                if i + 1 < m + 2:
                    y = i * (n + 1) + (j - 1)
                    dct[nums1[j] - nums1[j - 1]].append([x, y])
                if j + 1 < n + 2:
                    y = (i - 1) * (n + 1) + j
                    dct[nums2[i] - nums2[i - 1]].append([x, y])
        ans = 0
        uf = UnionFind((n + 1) * (m + 1))
        for c in sorted(dct):
            for i, j in dct[c]:
                if uf.union(i, j):
                    ans += c
        ac.st(ans)
        return

    @staticmethod
    def lc_1584_1(nums: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/min-cost-to-connect-all-points/
        tag: dense_graph|prim|mst
        """

        # primmst|，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = abs(x1 - x2) + abs(y1 - y2)
            return res

        n = len(nums)
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        return ans

    @staticmethod
    def lc_1584_2(nums: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/min-cost-to-connect-all-points/
        tag: dense_graph|prim|mst
        """

        # primmst|，适合稠密图场景
        def dis(xx1, yy1, xx2, yy2):
            res = abs(xx1 - xx2) + abs(yy1 - yy2)
            return res

        n = len(nums)
        edges = []
        for i in range(n):
            x1, y1 = nums[i]
            for j in range(i + 1, n):
                x2, y2 = nums[j]
                edges.append([i, j, dis(x1, y1, x2, y2)])

        tree = MinimumSpanningTree(edges, n, "prim")
        return tree.cost

    @staticmethod
    def lg_p1556(ac=FastIO()):
        # mst|，增|虚拟源点
        n = ac.read_int()
        edges = []
        for i in range(n):
            w = ac.read_int()
            edges.append([0, i + 1, w])
            # 虚拟源点
        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i + 1, n):
                edges.append([i + 1, j + 1, grid[j]])
        # kruskalmst|
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
    def abc_65d(ac=FastIO()):
        # mst|变形问题
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it][0])
        edges = []
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][0] - nums[x][0]
            edges.append([x, y, d])
        uf = UnionFind(n)
        ind.sort(key=lambda it: nums[it][1])
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][1] - nums[x][1]
            edges.append([x, y, d])
        edges.sort(key=lambda it: it[2])
        ans = 0
        for i, j, d in edges:
            if uf.union(i, j):
                ans += d
        ac.st(ans)
        return

    @staticmethod
    def ac_3728(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3731/
        tag: prim|mst|dense_graph|specific_plan
        """

        # primmst|，适合稠密图场景，并获取具体连边specific_plan，也可直接Kruskal（超时）

        def dis(aa, bb):
            if aa == 0:
                return cost[bb]
            if bb == 0:
                return cost[aa]

            return (k[aa] + k[bb]) * (abs(nums[aa][0] - nums[bb][0]) + abs(nums[aa][1] - nums[bb][1]))

        n = ac.read_int()
        nums = [[inf, inf]] + [ac.read_list_ints() for _ in range(n)]
        cost = [inf] + ac.read_list_ints()
        k = [inf] + ac.read_list_ints()

        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(n + 1)))
        visit = [inf] * (n + 1)
        visit[nex] = 0
        pre = [-1] * (n + 1)  # 记录mst|的父节点
        edge = []
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            for j in rest:
                dj = dis(i, j)
                if dj < visit[j]:
                    visit[j] = dj
                    pre[j] = i
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
            if nex != -1:
                edge.append([pre[nex], nex])
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st(ans if ans < inf else -1)
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


class DistanceLimitedPathsExist:
    # LC1724
    def __init__(self, n: int, edge_list: List[List[int]]):
        uf = UnionFind(n)
        edge = []
        for i, j, d in sorted(edge_list, key=lambda it: it[-1]):
            if uf.union(i, j):
                edge.append([i, j, d])

        self.nodes = []
        part = uf.get_root_part()
        self.root = [0] * n
        for p in part:
            self.nodes.append(part[p])
            i = len(self.nodes) - 1
            for x in part[p]:
                self.root[x] = i
        self.ind = [{num: i for i, num in enumerate(node)} for node in self.nodes]
        dct = [[dict() for _ in range(len(node))] for node in self.nodes]

        for i, j, d in edge:
            r = self.root[i]
            dct[r][self.ind[r][i]][self.ind[r][j]] = d
            dct[r][self.ind[r][j]][self.ind[r][i]] = d
        # 倍增维护查询任意两点路径的最大边权值
        self.tree = [TreeAncestorWeightSecond(dc) for dc in dct]

    def query(self, p: int, q: int, limit: int) -> bool:
        if self.root[p] != self.root[q]:
            return False
        r = self.root[p]
        i = self.ind[r][p]
        j = self.ind[r][q]
        return self.tree[r].get_dist_weight_max_second(i, j)[0] < limit
