"""

Algorithm：floyd|several_source_shortest_path|undirected_graph|directed_graph|pos_weight|neg_weight|negative_circle|shortest_circle
Description：shortest_path|longest_path|necessary_point_on_shortest_path|necessary_point_on_longest_path|necessary_edge
specific_plan： floyd need dp[i][j] where pre[i][j] = k, and bellman-ford dijkstra need pre[v] = u

====================================LeetCode====================================
2642（https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/）floyd|dynamic_graph|shortest_path
1462（https://leetcode.cn/problems/course-schedule-iv/）transitive_closure|floyd

=====================================LuoGu======================================
P1119（https://www.luogu.com.cn/problem/P1119）offline_query|floyd|dynamic_graph
P1476（https://www.luogu.com.cn/problem/P1476）floyd|longest_path|specific_plan
P3906（https://www.luogu.com.cn/problem/P3906）floyd|shortest_path|specific_plan

P2009（https://www.luogu.com.cn/problem/P2009）floyd|shortest_path
P2419（https://www.luogu.com.cn/problem/P2419）floyd|topological_sort
P2910（https://www.luogu.com.cn/problem/P2910）shortest_path|floyd
P6464（https://www.luogu.com.cn/problem/P6464）brute_force|floyd|dynamic_graph
P6175（https://www.luogu.com.cn/problem/P6175）floyd|brute_force|O(n^3)|bfs|dijkstra
B3611（https://www.luogu.com.cn/problem/B3611）transitive_closure|floyd
P1613（https://www.luogu.com.cn/problem/P1613）floyd|several_floyd|shortest_path
P8312（https://www.luogu.com.cn/problem/P8312）limited_floyd|shortest_path|several_floyd
P8794（https://www.luogu.com.cn/problem/P8794）binary_search|floyd


===================================CodeForces===================================
472D（https://codeforces.com/problemset/problem/472/D）floyd|construction|shortest_path
1205B（https://codeforces.com/problemset/problem/1205/B）data_range|floyd|undirected_shortest_circle
25C（https://codeforces.com/problemset/problem/25/C）floyd
543B（https://codeforces.com/problemset/problem/543/B）bfs|brute_force|observation|floyd|implemention


====================================AtCoder=====================================
ABC051D（https://atcoder.jp/contests/abc051/tasks/abc051_d）floyd|shortest_path|necessary_edge|classical
ABC074D（https://atcoder.jp/contests/abc074/tasks/arc083_b）shortest_path_spanning_tree|floyd|dynamic_graph
ABC143E（https://atcoder.jp/contests/abc143/tasks/abc143_e）floyd|build_graph|shortest_path|several_floyd
ABC286E（https://atcoder.jp/contests/abc286/tasks/abc286_e）floyd|classical
ABC243E（https://atcoder.jp/contests/abc243/tasks/abc243_e）get_cnt_of_shortest_path|undirected|dijkstra|floyd|classical
ABC208D（https://atcoder.jp/contests/abc208/tasks/abc208_d）floyd|shortest_path|classical
ABC369E（https://atcoder.jp/contests/abc369/tasks/abc369_e）floyd|permutation|brute_force
ABC375F（https://atcoder.jp/contests/abc375/tasks/abc375_f）floyd|add_undirected_edge
        
=====================================AcWing=====================================
4872（https://www.acwing.com/problem/content/submission/4875/）floyd|reverse_thinking|shortest_path|reverse_graph

"""
import math
import random
from collections import defaultdict
from heapq import heappop, heappush
from itertools import permutations
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.graph.floyd.template import WeightedGraphForFloyd
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1613(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1613
        tag: floyd|several_floyd|shortest_path
        """
        n, m = ac.read_list_ints()

        pre = [0] * n * n
        cur = [0] * n * n
        dis = [math.inf] * n * n
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            pre[u * n + v] = dis[u * n + v] = 1

        for x in range(1, 32):
            for k in range(n):
                for i in range(n):
                    if not pre[i * n + k]:
                        continue
                    for j in range(n):
                        if pre[i * n + k] and pre[k * n + j]:
                            cur[i * n + j] = dis[i * n + j] = 1

            for i in range(n * n):
                pre[i] = cur[i]
                cur[i] = 0

        for k in range(n):
            for i in range(n):
                if dis[i * n + k] == math.inf:
                    continue
                for j in range(n):
                    dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        ac.st(dis[n - 1])
        return

    @staticmethod
    def ac_4872(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4875/
        tag: floyd|reverse_thinking|shortest_path|reverse_graph
        """
        n = ac.read_int()
        dp = [ac.read_list_ints() for _ in range(n)]
        a = ac.read_list_ints_minus_one()
        node = []
        ans = []
        for ind in range(n - 1, -1, -1):
            x = a[ind]
            node.append(x)
            cur = 0
            for i in node:
                for j in node:
                    dp[i][x] = min(dp[i][x], dp[i][j] + dp[j][x])
                    dp[x][i] = min(dp[x][i], dp[x][j] + dp[j][i])

            for i in node:
                for j in node:
                    dp[i][j] = min(dp[i][j], dp[i][x] + dp[x][j])
                    cur += dp[i][j]
            ans.append(cur)

        ac.lst(ans[::-1])
        return

    @staticmethod
    def lg_p1119(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1119
        tag: offline_query|floyd|dynamic_graph|undirected
        """
        n, m = ac.read_list_ints()
        repair = ac.read_list_ints()
        inf = 10 ** 10
        graph = WeightedGraphForFloyd(n, inf)
        for i in range(m):
            i, j, w = ac.read_list_ints()
            graph.add_undirected_edge_initial(i, j, w)

        k = 0
        for _ in range(ac.read_int()):
            x, y, t = ac.read_list_ints()
            while k < n and repair[k] <= t:
                graph.update_point_undirected(k)
                k += 1
            if graph.dis[x * n + y] < inf and x < k and y < k:
                ac.st(graph.dis[x * n + y])
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p1476(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1476
        tag: floyd|longest_path|specific_plan|classical
        """
        n = ac.read_int() + 1
        m = ac.read_int()
        inf = 10 ** 6
        graph = WeightedGraphForFloyd(n, inf)
        for _ in range(m):
            i, j, k = ac.read_list_ints_minus_one()
            if i == j:
                continue
            graph.add_directed_edge_initial(i, j, -k - 1)
        graph.initialize_directed()

        ans = -graph.dis[n - 1]
        path = graph.get_nodes_between_src_and_dst(0, n - 1)
        ac.st(ans)
        ac.lst([x + 1 for x in path])
        return

    @staticmethod
    def lg_p3906(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3906
        tag: floyd|shortest_path|specific_plan|classical
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 4
        graph = WeightedGraphForFloyd(n, inf)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge_initial(i, j, 1)
        graph.initialize_undirected()

        for _ in range(ac.read_int()):
            i, j = ac.read_list_ints_minus_one()
            path = graph.get_nodes_between_src_and_dst(i, j)
            ac.lst([x + 1 for x in path])
        return

    @staticmethod
    def lg_b3611(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/B3611
        tag: transitive_closure|floyd
        """
        n = ac.read_int()
        inf = 10 ** 5
        graph = WeightedGraphForFloyd(n, inf)
        dp = []
        for _ in range(n):
            dp.extend(ac.read_list_ints())
        for i in range(n):
            for j in range(n):
                if dp[i * n + j]:
                    graph.add_directed_edge_initial(i, j, 1)
        for i in range(n):
            graph.dis[i * n + i] = inf

        graph.initialize_directed()
        for i in range(n):
            ac.lst([int(graph.dis[i * n + j] < inf) for j in range(n)])
        return

    @staticmethod
    def abc_051d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc051/tasks/abc051_d
        tag: floyd|shortest_path|necessary_edge|classical|reverse_thinking
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 7
        graph = WeightedGraphForFloyd(n, inf)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        for i, j, w in edges:
            graph.add_undirected_edge_initial(i, j, w + 1)
        graph.initialize_undirected()
        ans = sum(graph.dis[i * n + j] < w + 1 for i, j, w in edges)
        ac.st(ans)
        return

    @staticmethod
    def abc_074d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc074/tasks/arc083_b
        tag: shortest_path_spanning_tree|floyd|dynamic_graph
        """
        n = ac.read_int()
        grid = []
        for _ in range(n):
            grid.extend(ac.read_list_ints())

        vals = []
        for i in range(n):
            if grid[i * n + i]:
                ac.st(-1)
                return
            for j in range(i + 1, n):
                if grid[i * n + j] != grid[j * n + i]:
                    ac.st(-1)
                    return
                vals.append(grid[i * n + j] * n * n + i * n + j)

        vals.sort()
        inf = 10 ** 15
        graph = WeightedGraphForFloyd(n, inf)
        ans = 0
        for val in vals:
            w, num = val // n // n, val % (n * n)
            i, j = num // n, num % n
            if graph.dis[i * n + j] < w:
                ac.st(-1)
                return
            if graph.dis[i * n + j] == w:
                continue
            ans += w
            graph.add_undirected_edge(i, j, w)
        ac.st(ans)
        return

    @staticmethod
    def abc_143e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc143/tasks/abc143_e
        tag: floyd|build_graph|shortest_path|several_floyd
        """
        n, m, ll = ac.read_list_ints()

        inf = 10 ** 15
        graph = WeightedGraphForFloyd(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            graph.add_undirected_edge_initial(i, j, w)

        graph.initialize_undirected()
        for i in range(n):
            for j in range(i + 1, n):
                graph.dis[i * n + j] = 0 if graph.dis[j * n + i] <= ll else inf

        for k in range(n):
            for i in range(graph.n):
                if graph.dis[i * graph.n + k] == graph.inf:
                    continue
                for j in range(i + 1, graph.n):
                    cur = graph.dis[i * graph.n + k] + graph.dis[k * graph.n + j] + 1
                    graph.dis[i * graph.n + j] = graph.dis[j * graph.n + i] = min(graph.dis[i * graph.n + j], cur)

        for _ in range(ac.read_int()):
            i, j = ac.read_list_ints_minus_one()
            ans = graph.dis[i * n + j]
            ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def cf_472d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/472/D
        tag: floyd|construction|shortest_path
        """

        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            if grid[i][i]:
                ac.no()
                return
            for j in range(i + 1, n):
                if grid[i][j] != grid[j][i] or not grid[i][j]:
                    ac.no()
                    return
        if n == 1:
            ac.yes()
            return
        for i in range(n):
            j = 1 if not i else 0
            for r in range(n):
                if grid[i][r] < grid[i][j] and i != r:
                    j = r
            for k in range(n):
                if abs(grid[i][k] - grid[j][k]) != grid[i][j]:
                    ac.no()
                    return
        ac.yes()
        return

    @staticmethod
    def lg_p8312(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8312
        tag: limited_floyd|shortest_path|several_floyd
        """
        n, m = ac.read_list_ints()
        dis = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0

        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dis[a][b] = min(dis[a][b], c)

        dct = [d[:] for d in dis]
        k, q = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(q)]
        k = min(k, n)
        for _ in range(k - 1):
            cur = [d[:] for d in dis]
            for p in range(n):
                for i in range(n):
                    for j in range(n):
                        cur[i][j] = min(cur[i][j], dis[i][p] + dct[p][j])
            dis = [d[:] for d in cur]

        for c, d in nums:
            res = dis[c][d]
            ac.st(res if res < math.inf else -1)
        return

    @staticmethod
    def lg_p8794(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8794
        tag: binary_search|floyd
        """

        def get_dijkstra_result_mat(mat: List[List[int]], src: int) -> List[float]:
            len(mat)
            dis = [math.inf] * n
            stack = [[0, src]]
            dis[src] = 0
            visit = set(list(range(n)))
            while stack:
                d, ii = heappop(stack)
                if dis[ii] < d:
                    continue
                visit.discard(ii)
                for j in visit:
                    dj = mat[ii][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, [dj, j])
            return dis

        n, q = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(n)]
        lower = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(lower, i))
        if ans > q:
            ac.st(-1)
            return

        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(grid, i))
        if ans <= q:
            ac.st(0)
            return

        def check(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            dis = 0
            for y in range(n):
                dis += sum(get_dijkstra_result_mat(cur, y))
            return dis <= q

        def check2(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            for k in range(n):
                for a in range(n):
                    for b in range(a + 1, n):
                        cur[a][b] = cur[b][a] = min(cur[a][b], cur[a][k] + cur[k][b])
            return sum(sum(c) for c in cur) <= q

        low = 1
        high = n * 10 ** 5
        BinarySearch().find_int_left(low, high, check)
        ans = BinarySearch().find_int_left(low, high, check2)
        ac.st(ans)
        return

    @staticmethod
    def lc_2642():
        """
        url: https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/
        tag: floyd|dynamic_graph|shortest_path
        """

        class Graph:

            def __init__(self, n: int, edges: List[List[int]]):
                self.graph = WeightedGraphForFloyd(n, 10 ** 15)
                for i, j, w in edges:
                    self.graph.add_directed_edge_initial(i, j, w)
                self.graph.initialize_directed()

            def add_edge(self, edge: List[int]) -> None:
                i, j, w = edge
                if w < self.graph.dis[i * self.graph.n + j]:
                    self.graph.add_directed_edge(i, j, w)

            def shortest_path(self, node1: int, node2: int) -> int:
                ans = self.graph.dis[node1 * self.graph.n + node2]
                return ans if ans < self.graph.inf else -1

        return Graph

    @staticmethod
    def abc_286e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc286/tasks/abc286_e
        tag: floyd|classical
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        s = [ac.read_str() for _ in range(n)]

        inf = 10 ** 18
        dis = [inf] * n * n
        gain = [0] * n * n
        for i in range(n):
            dis[i * n + i] = 0
            gain[i * n + i] = a[i]

        for i in range(n):
            for j in range(n):
                if s[i][j] == "Y":
                    dis[i * n + j] = 1
                    gain[i * n + j] = a[i] + a[j]

        for k in range(n):
            for i in range(n):
                if dis[i * n + k] == inf:
                    continue
                for j in range(n):
                    cur = dis[i * n + k] + dis[k * n + j]
                    g = gain[i * n + k] + gain[k * n + j] - a[k]
                    if cur < dis[i * n + j] or (cur == dis[i * n + j] and g > gain[i * n + j]):
                        gain[i * n + j] = g
                        dis[i * n + j] = cur

        for _ in range(ac.read_int()):
            u, v = ac.read_list_ints_minus_one()
            if dis[u * n + v] == inf:
                ac.st("Impossible")
            else:
                ac.lst([dis[u * n + v], gain[u * n + v]])
        return

    @staticmethod
    def abc_243e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc243/tasks/abc243_e
        tag: get_cnt_of_shortest_path|undirected|dijkstra|floyd|classical
        """
        n, m = ac.read_list_ints()
        graph = WeightedGraphForFloyd(n)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        for i, j, w in edges:
            graph.add_undirected_edge_initial(i, j, w + 1)
        mod = random.getrandbits(32)
        graph.get_cnt_of_shortest_path_undirected(mod)
        ans = sum(graph.cnt[i * n + j] > 1 or graph.dis[i * n + j] < w + 1 for i, j, w in edges)
        ac.st(ans)
        return

    @staticmethod
    def abc_208d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc208/tasks/abc208_d
        tag: floyd|shortest_path|classical
        """
        n, m = ac.read_list_ints()
        dp = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        for _ in range(m):
            i, j, c = ac.read_list_ints_minus_one()
            c += 1
            dp[i][j] = c
        ans = 0
        tot = sum(sum(x if x < math.inf else 0 for x in dp[i]) for i in range(n))
        for i in range(n):
            for a in range(n):
                if dp[a][i] < math.inf:
                    for b in range(n):
                        if dp[a][i] + dp[i][b] < dp[a][b]:
                            if dp[a][b] < math.inf:
                                tot -= dp[a][b]
                            dp[a][b] = dp[a][i] + dp[i][b]
                            tot += dp[a][b]
            ans += tot
        ac.st(ans)
        return

    @staticmethod
    def cf_1205b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1205/B
        tag: data_range|floyd|undirected_shortest_circle
        """
        ac.read_int()
        nums = ac.read_list_ints()
        nums = [num for num in nums if num]
        n = len(nums)
        if n <= 2:
            ac.st(-1)
            return
        dct = [[] for _ in range(64)]
        for i, num in enumerate(nums):
            for x in range(64):
                if (num >> x) & 1:
                    dct[x].append(i)
        if any(len(dct[w]) >= 3 for w in range(64)):
            ac.st(3)
            return

        assert n <= 200
        edge = [math.inf] * n * n
        dis = [math.inf] * n * n
        for x in range(64):
            if len(dct[x]) == 2:
                i, j = dct[x][0], dct[x][1]
                dis[i * n + j] = dis[j * n + i] = 1
                edge[i * n + j] = edge[j * n + i] = 1
        ans = math.inf
        for k in range(n):
            for i in range(n):
                if dis[i * n + k] == math.inf:
                    continue
                for j in range(i + 1, n):
                    ans = min(ans, dis[i * n + j] + edge[i * n + k] + edge[k * n + j])  # classical
            for i in range(n):
                if dis[i * n + k] == math.inf:
                    continue
                for j in range(i + 1, n):
                    dis[j * n + i] = dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def abc_369e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc369/tasks/abc369_e
        tag: floyd|permutation|brute_force
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 18
        graph = WeightedGraphForFloyd(n, inf)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        for i, j, w in edges:
            graph.add_undirected_edge_initial(i, j, w + 1)
        graph.initialize_undirected()

        for _ in range(ac.read_int()):
            k = ac.read_int()
            lst = ac.read_list_ints_minus_one()
            ans = inf
            cost = sum(edges[x][-1] + 1 for x in lst)

            for item in permutations(lst, k):

                pre = defaultdict(lambda: inf)
                pre[0] = cost
                for x in item:
                    cur = defaultdict(lambda: inf)
                    for p in pre:
                        cur[edges[x][0]] = min(cur[edges[x][0]], pre[p] + graph.dis[p * graph.n + edges[x][1]])
                        cur[edges[x][1]] = min(cur[edges[x][1]], pre[p] + graph.dis[p * graph.n + edges[x][0]])
                    pre = cur
                ans = min(ans, min(pre[p] + graph.dis[p * graph.n + graph.n - 1] for p in pre))
            ac.st(ans)
        return

    @staticmethod
    def abc_375f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc375/tasks/abc375_f
        tag: floyd|add_undirected_edge
        """
        n, m, q = ac.read_list_ints()
        math.inf = 2 * 10 ** 15
        graph = WeightedGraphForFloyd(n, math.inf)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        visit = [1] * m
        queries = [ac.read_list_ints_minus_one() for _ in range(q)]
        for lst in queries:
            if lst[0] == 0:
                visit[lst[1]] = 0

        for x in range(m):
            if visit[x]:
                i, j, w = edges[x]
                w += 1
                graph.add_undirected_edge_initial(i, j, w)
        graph.initialize_undirected()

        res = []
        for x in range(q - 1, -1, -1):
            lst = queries[x]
            if lst[0] == 0:
                ind = lst[1]
                i, j, w = edges[ind]
                w += 1
                graph.add_undirected_edge(i, j, w)
            else:
                i, j = lst[1], lst[2]
                res.append(graph.dis[i * n + j])
        res.reverse()
        for x in res:
            ac.st(x if x < math.inf else -1)
        return
