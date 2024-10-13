"""
Algorithm：dijkstra|strictly_second_shortest_path|longest_path|shortest_path_spanning_tree|two_params_dijkstra
Description：limited_shortest_path|layered_dijkstra|directed_smallest_circle|undirected_smallest_circle|shortest_path_mst

====================================LeetCode====================================
42（https://leetcode.cn/problems/trapping-rain-water/）prefix_suffix
407（https://leetcode.cn/problems/trapping-rain-water-ii/）maximum_weight_on_shortest_path
787（https://leetcode.cn/problems/cheapest-flights-within-k-stops/）limited_shortest_path
1293（https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination/）limited_shortest_path
2203（https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/）several_dijkstra|shortest_path
2258（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）bfs|preprocess|shortest_path|maximum_weight_on_shortest_path
2290（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）shortest_path
499（https://leetcode.cn/problems/the-maze-iii/）two_params_dijkstra
6442（https://leetcode.cn/problems/modify-graph-edge-weights/）several_dijkstra|shortest_path|greedy
2714（https://leetcode.cn/problems/find-shortest-path-with-k-hops/）limited_shortest_path|layered_dijkstra
2699（https://leetcode.cn/problems/modify-graph-edge-weights/）dijkstra|shortest_path|greedy
1786（https://leetcode.cn/problems/number-of-restricted-paths-from-first-to-last-node/）dijkstra|limited_shortest_path|counter|dag|undirected_to_dag
1928（https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/）dijkstra|limited_shortest_path|floyd
75（https://leetcode.cn/problems/rdmXM7/）bfs|minimum_max_weight|shortest_path|maximum_weight_on_shortest_path
1976（https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/）dijkstra|number_of_shortest_path|classical
2045（https://leetcode.cn/problems/second-minimum-time-to-reach-destination/）strictly_second_shortest_path|classical
2093（https://leetcode.cn/problems/minimum-cost-to-reach-city-with-discounts/）dijkstra|limited_shortest_path
882（https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/description/）dijkstra
2577（https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/）dijkstra|matrix
2065（https://leetcode.cn/problems/maximum-path-quality-of-a-graph/）back_trace|dijkstra|shortest_path|prune
3112（https://leetcode.com/problems/minimum-time-to-visit-disappearing-nodes/description/）dijkstra|template|classical

=====================================LuoGu======================================
P3371（https://www.luogu.com.cn/problem/P3371）shortest_path
P4779（https://www.luogu.com.cn/problem/P4779）shortest_path
P1629（https://www.luogu.com.cn/problem/P1629）shortest_path|several_dijkstra
P1462（https://www.luogu.com.cn/problem/P1462）limited_shortest_path
P1339（https://www.luogu.com.cn/problem/P1339）shortest_path
P1342（https://www.luogu.com.cn/problem/P1342）shortest_path|several_dijkstra|reverse_graph|reverse_dijkstra
P1576（https://www.luogu.com.cn/problem/P1576）heapq|pos_to_neg|shortest_path
P1821（https://www.luogu.com.cn/problem/P1821）shortest_path|several_dijkstra|reverse_graph|reverse_dijkstra
P1882（https://www.luogu.com.cn/problem/P1882）shortest_path
P1907（https://www.luogu.com.cn/problem/P1907）build_graph|shortest_path
P1744（https://www.luogu.com.cn/problem/P1744）shortest_path
P1529（https://www.luogu.com.cn/problem/P1529）shortest_path
P1649（https://www.luogu.com.cn/problem/P1649）define_distance|shortest_path
P2083（https://www.luogu.com.cn/problem/P2083）reverse_graph|shortest_path
P2299（https://www.luogu.com.cn/problem/P2299）shortest_path
P2683（https://www.luogu.com.cn/problem/P2683）shortest_path|union_find
P1396（https://www.luogu.com.cn/problem/P1396）shortest_path|maximum_weight_on_shortest_path
P1346（https://www.luogu.com.cn/problem/P1346）build_graph|shortest_path
list?user=739032&status=12&page=11（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）shortest_path
P2784（https://www.luogu.com.cn/problem/P2784）maximum_mul_path
P1318（https://www.luogu.com.cn/problem/P1318）prefix_suffix
P2888（https://www.luogu.com.cn/problem/P2888）shortest_path|minimum_max_weight_on_shortest_path
P2935（https://www.luogu.com.cn/problem/P2935）shortest_path
P2951（https://www.luogu.com.cn/problem/P2951）shortest_path
P2984（https://www.luogu.com.cn/problem/P2984）shortest_path
P3003（https://www.luogu.com.cn/problem/P3003）several_dijkstra|shortest_path
P3094（https://www.luogu.com.cn/problem/P3094）shortest_path|preprocess
P3905（https://www.luogu.com.cn/problem/P3905）reverse_thinking|build_graph|shortest_path
P5764（https://www.luogu.com.cn/problem/P5764）several_dijkstra|shortest_path
P5767（https://www.luogu.com.cn/problem/P5767）build_graph|shortest_path
P6770（https://www.luogu.com.cn/problem/P6770）shortest_path
P6833（https://www.luogu.com.cn/problem/P6833）several_dijkstra|shortest_path|brute_force
P7551（https://www.luogu.com.cn/problem/P7551）shortest_path|multi_edge|self_loop
P6175（https://www.luogu.com.cn/problem/P6175）dijkstra|brute_force|dfs
P4568（https://www.luogu.com.cn/problem/P4568）build_graph|layer_dijkstra|shortest_path
P2865（https://www.luogu.com.cn/problem/P2865）strictly_second_shortest_path
P2622（https://www.luogu.com.cn/problem/P2622）state_compress|dijkstra|shortest_path
P1073（https://www.luogu.com.cn/problem/P1073）reverse_graph|build_graph|dijkstra
P1300（https://www.luogu.com.cn/problem/P1300）dijkstra|shortest_path
P1354（https://www.luogu.com.cn/problem/P1354）build_graph|dijkstra|shortest_path
P1608（https://www.luogu.com.cn/problem/P1608）dijkstra|undirected|directed|number_of_shortest_path
P1828（https://www.luogu.com.cn/problem/P1828）several_dijkstra|shortest_path
P2047（https://www.luogu.com.cn/problem/P2047）node_shortest_path_count|dijkstra|shortest_path|floyd|classical
P2269（https://www.luogu.com.cn/problem/P2269）shortest_path
P2349（https://www.luogu.com.cn/problem/P2349）shortest_path
P2914（https://www.luogu.com.cn/problem/P2914）dijkstra|build_graph|dynamic_graph
P3020（https://www.luogu.com.cn/problem/P3020）dijkstra|shortest_path
P3057（https://www.luogu.com.cn/problem/P3057）dijkstra|shortest_path
P3753（https://www.luogu.com.cn/problem/P3753）shortest_path|two_params
P3956（https://www.luogu.com.cn/problem/P3956）several_params|dijkstra
P4880（https://www.luogu.com.cn/problem/P4880）brute_force|dijkstra|shortest_path
P4943（https://www.luogu.com.cn/problem/P4943）brute_force|several_dijkstra|shortest_path
P5201（https://www.luogu.com.cn/problem/P5201）shortest_path_spanning_tree|build_graph|tree_dp
P5663（https://www.luogu.com.cn/problem/P5663）shortest_odd_path|shortest_even_path
P5683（https://www.luogu.com.cn/problem/P5683）several_dijkstra|shortest_path|brute_force
P5837（https://www.luogu.com.cn/problem/P5837）dijkstra|several_params
P5930（https://www.luogu.com.cn/problem/P5930）dijkstra|minimum_max_weight_on_shortest_path
P6063（https://www.luogu.com.cn/problem/P6063）dijkstra|minimum_max_weight_on_shortest_path
P6512（https://www.luogu.com.cn/problem/P6512）shortest_path
P8385（https://www.luogu.com.cn/problem/P8385）brain_teaser|build_graph|shortest_path
P8724（https://www.luogu.com.cn/problem/P8724）shortest_path|layer_dijkstra
P8802（https://www.luogu.com.cn/problem/P8802）dijkstra|define_weight
P2176（https://www.luogu.com.cn/problem/P2176）brute_force|shortest_path
P1807（https://www.luogu.com.cn/problem/P1807）dag|longest_path|dag_dp|topological_sort

===================================CodeForces===================================
20C（https://codeforces.com/problemset/problem/20/C）shortest_path|specific_plan
1343E（https://codeforces.com/problemset/problem/1343/E）several_bfs|shortest_path|greedy|brute_force
715B（https://codeforces.com/contest/715/problem/B）several_dijkstra|shortest_path|greedy|dynamic_graph
1433G（https://codeforces.com/contest/1433/problem/G）several_source_dijkstra|shortest_path|brute_force
1650G（https://codeforces.com/contest/1650/problem/G）dijkstra|shortest_path|strictly_second_shortest_path|counter|zero_one_bfs
1915G（https://codeforces.com/contest/1915/problem/G）shortest_path|limited_shortest_path|dijkstra
1196F（https://codeforces.com/contest/1196/problem/F）shortest_path|data_range|kth_shortest|brute_force|data_range
1741G（https://codeforces.com/contest/1741/problem/G）shortest_path|brute_force|state_dp
1846G（https://codeforces.com/contest/1846/problem/G）shortest_path
449B（https://codeforces.com/contest/449/problem/B）shortest_path|not_shortest_path_spanning_tree|union_find
1307D（https://codeforces.com/problemset/problem/1307/D）math|graph|shortest_path|observation|brute_force
938D（https://codeforces.com/problemset/problem/938/D）dijkstra|fake_source|build_graph|classical
1817B（https://codeforces.com/problemset/problem/1817/B）undirected_shortest_circle|brute_force
1473E（https://codeforces.com/problemset/problem/1473/E）layer_dijkstra|observation|classical|brain_teaser
545E（https://codeforces.com/problemset/problem/545/E）shortest_path_spanning_tree|minimum_weight|dijkstra|classical|shortest_path_mst|greedy
786B（https://codeforces.com/contest/786/problem/B）segment_tree_opt_build_graph|dijkstra|classical|weighted_graph
1076D（https://codeforces.com/problemset/problem/1076/D）weighted_graph|shortest_path_mst

====================================AtCoder=====================================
ABC142F（https://atcoder.jp/contests/abc142/tasks/abc142_f）directed|directed_smallest_circle
ABC342E（https://atcoder.jp/contests/abc342/tasks/abc342_e）classical|dijkstra|longest_path
ABC325E（https://atcoder.jp/contests/abc325/tasks/abc325_e）classical|data_range
ABC305E（https://atcoder.jp/contests/abc305/tasks/abc305_e）dijkstra|classical|several_source|shortest_path
ABC271E（https://atcoder.jp/contests/abc271/tasks/abc271_e）shortest_path|brain_teaser|implemention
ABC348D（https://atcoder.jp/contests/abc348/tasks/abc348_d）bfs|dijkstra|limited_shortest_path|state|classical
ABC257F（https://atcoder.jp/contests/abc257/tasks/abc257_f）shortest_path|brute_force|bfs|classical
ABC252E（https://atcoder.jp/contests/abc252/tasks/abc252_e）shortest_path_spanning_tree|dijkstra|classical
ABC245G（https://atcoder.jp/contests/abc245/tasks/abc245_g）shortest_path|second_shortest_path|dijkstra|brain_teaser|classical
ABC237E（https://atcoder.jp/contests/abc237/tasks/abc237_e）dijkstra|negative_weight|graph_mapping|brain_teaser|classical
ABC211D（https://atcoder.jp/contests/abc211/tasks/abc211_d）dijkstra|get_cnt_of_shortest_path
ABC204E（https://atcoder.jp/contests/abc204/tasks/abc204_e）dijkstra|shortest_path|classical|observation
ABC375G（https://atcoder.jp/contests/abc375/tasks/abc375_g）dijkstra_for_cnt_of_shortest_path|key_edge_in_shortest_path

=====================================AcWing=====================================
176（https://www.acwing.com/problem/content/178/）dijkstra|implemention
3628（https://www.acwing.com/problem/content/3631/）shortest_path_spanning_tree
3772（https://www.acwing.com/problem/content/description/3775/）build_graph|reverse_graph|dijkstra|shortest_path|counter|greedy|implemention
3797（https://www.acwing.com/problem/content/description/3800/）shortest_path|brute_force|sort|greedy
4196（https://www.acwing.com/problem/content/4199/）shortest_path
385（https://www.acwing.com/problem/content/description/385/）dijkstra_for_cnt_of_strictly_second_shortest_path

================================LibraryChecker====================================
Shortest Path（https://judge.yosupo.jp/problem/shortest_path）shortest_path|specific_plan

"""
import math
import random
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop, heapify
from itertools import accumulate, permutations
from operator import add
from typing import List

from src.data_structure.segment_tree.template import SegmentTreeOptBuildGraphZKW
from src.graph.dijkstra.template import UnDirectedShortestCycle, LimitedWeightedGraph, \
    WeightedGraphForShortestPathMST, WeightedGraphForDijkstra
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p6175_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6175
        tag: dijkstra|brute_force|dfs
        """
        n, m = ac.read_list_ints()
        dct = [defaultdict(lambda: math.inf) for _ in range(n)]
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = min(dct[i - 1][j - 1], w)
            dct[j - 1][i - 1] = min(dct[j - 1][i - 1], w)
        for i in range(n):
            for j in dct[i]:
                if j > i:
                    edges.append([i, j, dct[i][j]])
        ans = UnDirectedShortestCycle.find_shortest_cycle_with_edge(n, dct, edges)
        ac.st(ans if ans != -1 else "No solution.")
        return

    @staticmethod
    def lg_p6175_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6175
        tag: dijkstra|brute_force|dfs
        """
        n, m = ac.read_list_ints()
        dct = [defaultdict(lambda: math.inf) for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = min(dct[i - 1][j - 1], w)
            dct[j - 1][i - 1] = min(dct[j - 1][i - 1], w)
        ans = UnDirectedShortestCycle().find_shortest_cycle_with_node(n, dct)
        ac.st(ans if ans != -1 else "No solution.")
        return

    @staticmethod
    def cf_1343e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1343/E
        tag: several_bfs|shortest_path|greedy|brute_force
        """
        for _ in range(ac.read_int()):
            n, m, a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            c -= 1
            prices = sorted(ac.read_list_ints())
            prices = list(accumulate(prices, add, initial=0))

            inf = m + 1
            graph = WeightedGraphForDijkstra(n, inf)
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)

            dis_a = graph.bfs_for_shortest_path(a)
            dis_b = graph.bfs_for_shortest_path(b)
            dis_c = graph.bfs_for_shortest_path(c)
            ans = math.inf
            for x in range(n):
                up = dis_b[x]
                down = dis_a[x] + dis_c[x]
                if up + down <= m:
                    cur = prices[up] + prices[up + down]
                    ans = min(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def cf_1650g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1650/problem/G
        tag: dijkstra|shortest_path|strictly_second_shortest_path|counter|zero_one_bfs
        """
        mod = 10 ** 9 + 7
        for _ in range(ac.read_int()):
            ac.read_str()
            n, m = ac.read_list_ints()
            s, t = ac.read_list_ints_minus_one()
            inf = 10 ** 9
            graph = WeightedGraphForDijkstra(n, inf)
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)

            dis, cnt = graph.bfs_for_cnt_of_strictly_second_shortest_path(s, 0, mod)
            ans = cnt[t * 2]
            if dis[t * 2 + 1] == dis[t * 2] + 1:
                ans += cnt[t * 2 + 1]
                ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def ac_385(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/385/
        tag: dijkstra_for_cnt_of_strictly_second_shortest_path
        """

        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            inf = 10 ** 18
            graph = WeightedGraphForDijkstra(n, inf)
            for _ in range(m):
                i, j, w = ac.read_list_ints_minus_one()
                graph.add_directed_edge(i, j, w + 1)
            s, t = ac.read_list_ints_minus_one()
            dis, cnt = graph.dijkstra_for_cnt_of_strictly_second_shortest_path(s)
            ans = cnt[t * 2]
            if dis[t * 2 + 1] == dis[t * 2] + 1:
                ans += cnt[t * 2 + 1]
            ac.st(ans)
        return

    @staticmethod
    def lc_787(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/cheapest-flights-within-k-stops/
        tag: limited_shortest_path
        """
        dct = [[] for _ in range(n)]
        for u, v, p in flights:
            dct[u].append((v, p))

        stack = [(0, 0, src)]
        dis = [math.inf] * n
        while stack:
            cost, cnt, i = heappop(stack)
            if dis[i] <= cnt or cnt >= k + 2:
                continue
            if i == dst:
                return cost
            dis[i] = cnt
            for j, w in dct[i]:
                if cnt + 1 < dis[j]:
                    heappush(stack, (cost + w, cnt + 1, j))
        return -1

    @staticmethod
    def lc_2045(n: int, edges: List[List[int]], time: int, change: int) -> any:
        """
        url: https://leetcode.cn/problems/second-minimum-time-to-reach-destination/
        tag: strictly_second_shortest_path|classical
        """
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        src = 0
        dis = [[math.inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [(0, src)]
        while stack:
            d, i = heappop(stack)
            if dis[i][1] < d:
                continue
            for j in dct[i]:
                if (d // change) % 2 == 0:
                    nex_d = d + time
                else:
                    nex_d = (d // change + 1) * change + time
                if dis[j][0] > nex_d:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = nex_d
                    heappush(stack, (nex_d, j))
                elif dis[j][0] < nex_d < dis[j][1]:
                    dis[j][1] = nex_d
                    heappush(stack, (nex_d, j))
        return dis[-1][1]

    @staticmethod
    def lc_2065(values: List[int], edges: List[List[int]], max_time: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-path-quality-of-a-graph/
        tag: back_trace|dijkstra|shortest_path|prune|data_range
        """
        n = len(values)
        inf = math.inf
        graph = WeightedGraphForDijkstra(n, inf)

        for i, j, c in edges:
            graph.add_undirected_edge(i, j, c)

        dis = graph.dijkstra_for_shortest_path(0)
        stack = [[0, 0, {0}]]
        ans = 0
        visit = {tuple(sorted({0}) + [0]): 0}
        while stack:
            t, x, nodes = heappop(stack)
            if dis[x] + t <= max_time:
                cur = sum(values[j] for j in nodes)
                if cur > ans:
                    ans = cur
            ind = graph.point_head[x]
            while ind:
                y = graph.edge_to[ind]
                w = graph.edge_weight[ind]
                if t + w + dis[y] <= max_time:
                    state = tuple(sorted(nodes.union({y})) + [y])
                    if visit.get(state, math.inf) > t + w:
                        visit[state] = t + w
                        heappush(stack, [t + w, y, nodes.union({y})])
                ind = graph.edge_next[ind]
        return ans

    @staticmethod
    def lc_2093(n: int, highways: List[List[int]], discounts: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-cost-to-reach-city-with-discounts/
        tag: dijkstra|limited_shortest_path
        """
        dct = [[] for _ in range(n)]
        for u, v, p in highways:
            dct[u].append([v, p])
            dct[v].append([u, p])

        stack = [(0, 0, 0)]
        dis = [math.inf] * n
        while stack:
            cost, cnt, i = heappop(stack)
            if dis[i] <= cnt:
                continue
            if i == n - 1:
                return cost
            dis[i] = cnt
            for j, w in dct[i]:
                if cnt < dis[j]:
                    heappush(stack, (cost + w, cnt, j))
                if cnt + 1 < dis[j] and cnt + 1 <= discounts:
                    heappush(stack, (cost + w // 2, cnt + 1, j))

        return -1

    @staticmethod
    def lc_882(edges: List[List[int]], max_moves: int, n: int) -> int:
        """
        url: https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/description/
        tag: dijkstra
        """
        inf = math.inf
        graph = WeightedGraphForDijkstra(n, inf)

        for i, j, c in edges:
            graph.add_undirected_edge(i, j, c + 1)

        dis = graph.dijkstra_for_shortest_path(0)

        ans = sum(dis[i] <= max_moves for i in range(n))
        for i, j, c in edges:
            if c:
                if dis[i] <= max_moves:
                    a, b = max_moves - dis[i], c
                    left = a if a < b else b
                else:
                    left = 0

                if dis[j] <= max_moves:
                    a, b = max_moves - dis[j], c
                    right = a if a < b else b
                else:
                    right = 0

                if left + right <= c:
                    ans += left + right
                else:
                    ans += c
        return ans

    @staticmethod
    def lc_1293(grid: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination/
        tag: limited_shortest_path|classical|dijkstra_like
        """
        m, n = len(grid), len(grid[0])
        visit = defaultdict(lambda: float("math.inf"))

        stack = [[0, 0, 0, 0]]
        while stack:
            dis, cost, i, j = heappop(stack)
            if visit[(i, j)] <= cost or cost > k:
                continue
            if i == m - 1 and j == n - 1:
                return dis
            visit[(i, j)] = cost
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and cost + grid[x][y] < visit[(x, y)]:
                    heappush(stack, [dis + 1, cost + grid[x][y], x, y])
        return -1

    @staticmethod
    def lg_p1462(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1462
        tag: limited_shortest_path|classical
        """
        n, m, s = ac.read_list_ints()
        cost = [ac.read_int() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            if b not in dct[a] or dct[a][b] > c:
                dct[a][b] = c
            if a not in dct[b] or dct[b][a] > c:
                dct[b][a] = c

        visit = [0] * n
        stack = [(cost[0], 0, s)]
        while stack:
            dis, i, bd = heappop(stack)
            if visit[i] > bd:
                continue
            if i == n - 1:
                ac.st(dis)
                return
            visit[i] = bd
            for j in dct[i]:
                bj = bd - dct[i][j]
                if bj >= visit[j]:
                    visit[j] = bj
                    heappush(stack, (max(dis, cost[j]), j, bj))
        ac.st("AFK")
        return

    @staticmethod
    def lg_p4568(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4568
        tag: build_graph|layer_dijkstra|shortest_path|classical
        """
        n, m, k = ac.read_list_ints()
        s, t = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            dct[a].append((b, c))
            dct[b].append((a, c))

        n = len(dct)
        stack = [(0, 0, s)]
        dis = [math.inf] * n * (k + 1)
        dis[s * (k + 1)] = 0
        while stack:
            cost, cnt, i = heappop(stack)
            if dis[i * (k + 1) + cnt] < cost:
                continue
            if i == t:
                ac.st(cost)
                break
            for j, w in dct[i]:
                if dis[j * (k + 1) + cnt] > cost + w:
                    dis[j * (k + 1) + cnt] = cost + w
                    heappush(stack, (cost + w, cnt, j))
                if cnt + 1 <= k and dis[j * (k + 1) + cnt + 1] > cost:
                    dis[j * (k + 1) + cnt + 1] = cost
                    heappush(stack, (cost, cnt + 1, j))
        return

    @staticmethod
    def lg_p1629(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1629
        tag: shortest_path|several_dijkstra|reverse_thinking
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 18
        graph1 = WeightedGraphForDijkstra(n, inf)
        graph2 = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph1.add_directed_edge(i, j, w + 1)
            graph2.add_directed_edge(j, i, w + 1)
        dis1 = graph1.dijkstra_for_shortest_path(0)
        dis2 = graph2.dijkstra_for_shortest_path(0)
        ans = sum(dis1[i] + dis2[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2865(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2865
        tag: strictly_second_shortest_path
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 18
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        ans = graph.dijkstra_for_strictly_second_shortest_path(0)[(n - 1) * 2 + 1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p1807(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1807
        tag: dag|longest_path|dag_dp|topological_sort
        """
        n, m = ac.read_list_ints()
        inf = math.inf
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_directed_edge(i, j, w + 1)
        dis = graph.dijkstra_for_longest_path(0)
        ac.st(dis[-1] if dis[-1] > -inf else -1)
        return

    @staticmethod
    def lc_75(maze: List[str]) -> int:
        """
        url: https://leetcode.cn/problems/rdmXM7/
        tag: bfs|minimum_max_weight|shortest_path|maximum_weight_on_shortest_path
        """
        # shortest_path逃离
        m, n = len(maze), len(maze[0])
        start = [-1, -1]
        end = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = maze[i][j]
                if w == "S":
                    start = [i, j]
                elif w == "T":
                    end = [i, j]

        # 反向到达终点距离
        bfs = [[math.inf] * n for _ in range(m)]
        bfs[end[0]][end[1]] = 0
        stack = deque([end])
        while stack:
            i, j = stack.popleft()
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] != "#" and bfs[x][y] == math.inf:
                    bfs[x][y] = bfs[i][j] + 1
                    stack.append([x, y])

        # 魔法卷轴更新正向距离
        dis = [[math.inf] * n for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if bfs[i][j] < math.inf:
                    dis[i][j] = 0
                    if maze[i][j] == ".":
                        if maze[m - 1 - i][j] != "#":
                            dis[i][j] = max(dis[i][j], bfs[m - 1 - i][j])
                        if maze[i][n - 1 - j] != "#":
                            dis[i][j] = max(dis[i][j], bfs[i][n - 1 - j])

        # dijkstrashortest_path径边权最小的最大值
        visit = [[math.inf] * n for _ in range(n)]
        stack = [[dis[start[0]][start[1]], start[0], start[1]]]
        visit[start[0]][start[1]] = dis[start[0]][start[1]]
        while stack:
            d, i, j = heappop(stack)
            if visit[i][j] < d:
                continue
            visit[i][j] = d
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] != "#":
                    dj = max(d, dis[x][y])
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heappush(stack, [dj, x, y])
        x, y = end
        return visit[x][y] if visit[x][y] < math.inf else -1

    @staticmethod
    def lg_p2622(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2622
        tag: state_compress|dijkstra|shortest_path
        """
        # Dijkstra|状压shortest_path
        n = ac.read_int()
        m = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(m)]
        visit = [math.inf] * (1 << n)
        visit[(1 << n) - 1] = 0
        stack = [[0, (1 << n) - 1]]
        while stack:
            d, state = heappop(stack)
            if visit[state] < d:
                continue
            for i in range(m):
                cur = state
                for j in range(n):
                    if grid[i][j] == 1 and cur & (1 << j):
                        cur ^= (1 << j)
                    elif grid[i][j] == -1 and not cur & (1 << j):
                        cur ^= (1 << j)
                if d + 1 < visit[cur]:
                    visit[cur] = d + 1
                    heappush(stack, [d + 1, cur])
        ans = visit[0]
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p1073(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1073
        tag: reverse_graph|build_graph|dijkstra
        """
        # 正反两遍build_graph|，两个shortest_path
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        for _ in range(m):
            x, y, z = ac.read_list_ints_minus_one()
            dct[x].append(y)
            rev[y].append(x)
            if z == 1:
                dct[y].append(x)
                rev[x].append(y)

        # 前面最小值
        floor = [math.inf] * n
        stack = [[nums[0], 0]]
        floor[0] = nums[0]
        while stack:
            d, i = heappop(stack)
            if floor[i] < d:
                continue
            for j in dct[i]:
                dj = min(d, nums[j])
                if dj < floor[j]:
                    floor[j] = dj
                    heappush(stack, (dj, j))

        # 后面最大值
        ceil = [-math.inf] * n
        ceil[n - 1] = nums[n - 1]
        stack = [[-nums[n - 1], n - 1]]
        while stack:
            d, i = heappop(stack)
            if ceil[i] < d:
                continue
            for j in rev[i]:
                dj = max(-d, nums[j])
                if dj > ceil[j]:
                    ceil[j] = dj
                    heappush(stack, [-dj, j])
        ans = max(ceil[i] - floor[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p1300(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1300
        tag: dijkstra|shortest_path
        """
        # Dijkstra求shortest_path
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        ind = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dct = {"E": 0, "S": 1, "W": 2, "N": 3}
        start = [-1, -1]
        d = -1
        end = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = grid[i][j]
                if w in dct:
                    start = [i, j]
                    d = dct[w]
                if w == "F":
                    end = [i, j]

        dis = [[[math.inf] * 4 for _ in range(n)] for _ in range(m)]
        dis[start[0]][start[1]][d] = 0
        stack = [[0, start[0], start[1], d]]
        while stack:
            pre, i, j, d = heappop(stack)
            if dis[i][j][d] < pre:
                continue
            flag = False
            for cost, r in [[1, (d - 1) % 4], [5, (d + 1) % 4], [0, d]]:
                x, y = i + ind[r][0], j + ind[r][1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] != ".":
                    dj = pre + cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heappush(stack, [dj, x, y, r])
                        flag = True
            if not flag:
                cost, r = 10, (d + 2) % 4
                x, y = i + ind[r][0], j + ind[r][1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] != ".":
                    dj = pre + cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heappush(stack, [dj, x, y, r])
        ac.st(min(dis[end[0]][end[1]]))
        return

    @staticmethod
    def lg_p1354(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1354
        tag: build_graph|dijkstra|shortest_path
        """
        def dis(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        n = ac.read_int()
        nodes = [[0, 5], [10, 5]]
        line = []
        for _ in range(n):
            x, a1, a2, b1, b2 = [float(x) for x in ac.read_list_strs()]
            nodes.append([x, a1])
            nodes.append([x, a2])
            nodes.append([x, b1])
            nodes.append([x, b2])
            line.append([x, a1, a2, b1, b2])

        def check():
            for xx, aa1, aa2, bb1, bb2 in line:
                if left <= xx <= right:
                    if not (aa1<=k*xx+bb<=aa2) and not (bb1<=k*xx+bb<=bb2):
                        return False
            return True

        start = 0
        end = 1
        m = len(nodes)
        inf = math.inf
        graph = WeightedGraphForDijkstra(m, inf)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i]
                c, d = nodes[j]
                if a == c:
                    continue
                k = (d - b) / (c - a)
                bb = d - k * c
                left, right = min(a, c), max(a, c)
                if check():
                    x = dis(a, b, c, d)
                    graph.add_undirected_edge(i, j, x)
        ans = graph.dijkstra_for_shortest_path_float(start)[end]
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1608(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1608
        tag: dijkstra|undirected|directed|number_of_shortest_path
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 9
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            dct[i][j] = min(dct[i].get(j, inf), w+1)

        graph = WeightedGraphForDijkstra(n, inf)
        for i in range(n):
            for j, w in dct[i].items():
                graph.add_directed_edge(i, j, w)
            dct[i] = None
        del dct
        dis, cnt = graph.dijkstra_for_cnt_of_shortest_path(0)
        if dis[-1] == inf:
            ac.st("No answer")
        else:
            ac.lst([dis[-1], cnt[-1]])
        return

    @staticmethod
    def lg_p1828(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1828
        tag: several_dijkstra|shortest_path
        """
        n, p, c = ac.read_list_ints()
        cnt = Counter([ac.read_int() - 1 for _ in range(n)])
        inf = 10 ** 9
        graph = WeightedGraphForDijkstra(p, inf)
        for _ in range(c):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)

        total = [0] * p
        for i in cnt:
            dis = graph.dijkstra_for_shortest_path(i)
            for j in range(p):
                total[j] += dis[j] * cnt[i]
        ac.st(min(total))
        return

    @staticmethod
    def lg_p2047(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2047
        tag: node_shortest_path_count|dijkstra|shortest_path|floyd|classical
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 9
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        dis = []
        cnt = []
        for i in range(n):
            dd, cc = graph.dijkstra_for_cnt_of_shortest_path(i)
            dis.append(dd)
            cnt.append(cc)
        for i in range(n):
            ans = 0
            for j in range(n):
                for k in range(n):
                    if j != i and k != i:
                        if dis[j][i] + dis[i][k] == dis[j][k]:
                            ans += cnt[j][i] * cnt[i][k] / cnt[j][k]
            ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p2176(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2176
        tag: brute_force|shortest_path
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 18
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        path, dis = graph.dijkstra_for_shortest_path_from_src_to_dst(0, n - 1)

        ans = 0
        for x in range(m):
            graph.edge_weight[x * 2 + 1] *= 2
            graph.edge_weight[x * 2 + 2] *= 2
            _, cur = graph.dijkstra_for_shortest_path_from_src_to_dst(0, n - 1)
            ans = max(ans, cur - dis)
            graph.edge_weight[x * 2 + 1] //= 2
            graph.edge_weight[x * 2 + 2] //= 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p2269(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2269
        tag: shortest_path
        """
        # 比较两个维度的Dijkstra
        n, src, dst = ac.read_list_ints()
        src -= 1
        dst -= 1
        time = [ac.read_list_ints() for _ in range(n)]
        loss = [ac.read_list_floats() for _ in range(n)]

        # 丢失率与时延
        dis = [[math.inf, math.inf] for _ in range(n)]
        stack = [[0, 0, src]]
        dis[src] = [0, 0]
        # shortest_path
        while stack:
            ll, tt, i = heappop(stack)
            if dis[i] < [ll, tt]:
                continue
            if i == dst:
                break
            for j in range(n):
                if time[i][j] != -1 and loss[i][j] != -1:
                    nex_ll = 1 - (1 - ll) * (1 - loss[i][j])
                    nex_tt = tt + time[i][j]
                    if [nex_ll, nex_tt] < dis[j]:
                        dis[j] = [nex_ll, nex_tt]
                        heappush(stack, [nex_ll, nex_tt, j])
        res_ll = dis[dst][0]
        res_tt = dis[dst][1]
        ac.lst([res_tt, "%.4f" % res_ll])
        return

    @staticmethod
    def lg_p2349(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2349
        tag: shortest_path
        """
        # 比较两个项相|的shortest_path
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u][v] = min(dct[u].get(v, math.inf), w)
            dct[v][u] = min(dct[v].get(u, math.inf), w)

        # shortest_path模板
        dis = [math.inf] * n
        stack = [[0, 0, 0, 0]]
        dis[0] = 0
        while stack:
            dd, d, ceil, i = heappop(stack)
            if dis[i] < dd:
                continue
            if i == n - 1:
                break
            for j in dct[i]:
                dj = d + dct[i][j] + max(ceil, dct[i][j])
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj, d + dct[i][j], max(ceil, dct[i][j]), j])
        ac.st(dis[n - 1])
        return

    @staticmethod
    def lg_p2914(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2914
        tag: dijkstra|build_graph|dynamic_graph
        """

        # Dijkstra动态build_graph|距离

        def dis(x, y):
            if y in dct[x]:
                return 0
            x1, y1 = nums[x]
            x2, y2 = nums[y]
            res = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            return res if res <= m else math.inf

        n, w = ac.read_list_ints()
        m = ac.read_float()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [set() for _ in range(n)]
        for _ in range(w):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)

        n = len(dct)
        visit = [math.inf] * n
        stack = [[0, 0]]
        visit[0] = 0
        while stack:
            d, i = heappop(stack)
            if visit[i] < d:
                continue
            if i == n - 1:
                break
            for j in range(n):
                dj = dis(i, j) + d
                if dj < visit[j]:
                    visit[j] = dj
                    heappush(stack, (dj, j))
        ac.st(int(visit[-1] * 1000) if visit[-1] < math.inf else -1)
        return

    @staticmethod
    def lc_6442(n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/modify-graph-edge-weights/
        tag: several_dijkstra|shortest_path|greedy
        """
        dct = [[] for _ in range(n)]
        m = len(edges)
        book = [0] * m
        for ind, (i, j, w) in enumerate(edges):
            if w == -1:
                w = 1
                book[ind] = 1
                edges[ind][-1] = w
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍shortest_path最小情况下的距离
        dis0 = [math.inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            return []

        # 第二遍shortest_path
        dis1 = [math.inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是shortest_path上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）|上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def cf_715b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/715/problem/B
        tag: several_dijkstra|shortest_path|greedy|dynamic_graph
        """
        # 两遍shortest_path，greedy动态更新路径权值
        n, m, target, source, destination = ac.read_list_ints()
        edges = []
        dct = [[] for _ in range(n)]
        book = [0] * m
        for ind in range(m):
            i, j, w = ac.read_list_ints()
            if w == 0:
                w = 1
                book[ind] = 1
            edges.append([i, j, w])
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍shortest_path最小情况下的距离
        dis0 = [math.inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            ac.no()
            return

        # 第二遍shortest_path
        dis1 = [math.inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是shortest_path上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）|上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            ac.yes()
            for e in edges:
                ac.lst(e)
        else:
            ac.no()
        return

    @staticmethod
    def lg_p3753(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3753
        tag: shortest_path|two_params
        """
        # shortest_path变形两个维度的比较
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        cnt = 0
        for _ in range(m):
            x, y, s = ac.read_list_ints()
            cnt += s
            x -= 1
            y -= 1
            dct[x][y] = s
            dct[y][x] = s

        dis = [[math.inf, -math.inf] for _ in range(n)]
        stack = [[0, 0, 0]]
        dis[0] = [0, 0]

        while stack:
            dd, one, i = heappop(stack)
            if dis[i] < [dd, one]:
                continue
            for j in dct[i]:
                w = dct[i][j]
                dj = [dd + 1, one - w]
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj[0], dj[1], j])
        ans = cnt + dis[-1][1] + dis[-1][0] + dis[-1][1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p3956(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3956
        tag: several_params|dijkstra
        """
        # Dijkstra最小代价

        m, n = ac.read_list_ints()
        grid = [[-1] * m for _ in range(m)]
        for _ in range(n):
            x, y, c = ac.read_list_ints()
            grid[x - 1][y - 1] = c

        stack = [[0, 0, grid[0][0], 0, 0]]
        final = -1
        visit = defaultdict(lambda: math.inf)
        while stack:
            cost, magic, color, i, j = heappop(stack)
            if visit[(i, j, color)] <= cost:
                continue
            visit[(i, j, color)] = cost
            if i == j == m - 1:
                final = cost
                break
            for a, b in [[i - 1, j], [i, j + 1], [i, j - 1], [i + 1, j]]:
                if 0 <= a < m and 0 <= b < m:
                    if grid[i][j] != -1:
                        if grid[a][b] != -1:
                            heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
                        else:
                            heappush(stack, [cost + 2 + int(color != 0), 1, 0, a, b])
                            heappush(stack, [cost + 2 + int(color != 1), 1, 1, a, b])

                    else:
                        if grid[a][b] != -1:
                            heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
        ac.st(final)
        return

    @staticmethod
    def lg_p4880(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4880
        tag: brute_force|dijkstra|shortest_path
        """
        lst = []  # RE
        while True:
            cur = ac.read_list_ints()
            if not cur:
                break
            lst.extend(cur)
        lst.reverse()
        n, m, b, e = lst.pop(), lst.pop(), lst.pop(), lst.pop()
        b -= 1
        e -= 1
        inf = 10 ** 15
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            x, y, z = lst.pop(), lst.pop(), lst.pop()
            x -= 1
            y -= 1
            graph.add_undirected_edge(x, y, z)

        dis = graph.dijkstra_for_shortest_path(b)

        t = lst.pop()
        if not t:
            ac.st(dis[e])
            return

        nums = [[0, e + 1]] + [[lst.pop(), lst.pop()] for _ in range(t)]
        nums.sort()
        for i in range(t):
            pos = nums[i][1] - 1
            tt = nums[i + 1][0]
            if dis[pos] < tt:
                ac.st(max(dis[pos], nums[i][0]))
                return
        ac.st(max(dis[nums[-1][1]], nums[-1][0]))
        return

    @staticmethod
    def lg_p4943(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4943
        tag: brute_force|several_dijkstra|shortest_path
        """
        class Graph(WeightedGraphForDijkstra):
            def dijkstra_for_shortest_path_with_limit_and_target(self, src, initial, limit, target):
                dis = [self.inf] * self.n
                stack = [initial * self.n + src]
                dis[src] = initial
                while stack:
                    val = heappop(stack)
                    d, i = val // self.n, val % self.n
                    if i in target:
                        target.discard(i)
                    if dis[i] < d:
                        continue
                    ind = self.point_head[i]
                    while ind:
                        w = self.edge_weight[ind]
                        j = self.edge_to[ind]
                        if j not in limit:
                            dj = d + w
                            if dj < dis[j]:
                                dis[j] = dj
                                heappush(stack, dj * self.n + j)
                        ind = self.edge_next[ind]
                return dis

        n, m, k = ac.read_list_ints()
        if k:
            visit = set(ac.read_list_ints_minus_one())
        else:
            visit = set()
        inf = 10 ** 15
        graph = Graph(n, inf)
        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(a, b, c + 1)
        x, y = ac.read_list_ints_minus_one()

        dis1 = graph.dijkstra_for_shortest_path_with_limit_and_target(0, 0, visit, {0, x, y})
        dis11 = graph.dijkstra_for_shortest_path_with_limit_and_target(x, 0, visit, {0, x, y})
        dis2 = graph.dijkstra_for_shortest_path_with_limit_and_target(0, 0, set(), {0, x, y})
        dis22 = graph.dijkstra_for_shortest_path_with_limit_and_target(x, 0, set(), {0, x, y})

        ans = min(max(dis1[x], dis2[y]), max(dis1[y], dis2[x]),
                  dis1[x] + dis11[y], dis1[y] + dis11[y],
                  dis2[x] + dis22[y], dis2[y] + dis22[y])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5201(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5201
        tag: shortest_path_spanning_tree|build_graph|tree_dp
        """
        #  shortest_path_spanning_tree build_graph|，再tree_dp| 最优解
        n, m, t = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            dct[a].append([b, c + 1])
            dct[b].append([a, c + 1])
        for i in range(n):
            dct[i].sort()
        # 先跑一遍shortest_path
        dis = [math.inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))

        # 选择lexicographical_order较小的边建立shortest_path_spanning_tree
        edge = [[] for _ in range(n)]
        visit = [0] * n
        for i in range(n):
            for j, w in dct[i]:
                if visit[j]:
                    continue
                if dis[i] + w == dis[j]:
                    edge[i].append(j)
                    edge[j].append(i)
                    visit[j] = 1

        # tree_dp| 
        stack = [[0, -1]]
        ans = 0
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in edge[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                for j in edge[i]:
                    if j != fa:
                        nums[i] += nums[j]
                ans = max(ans, nums[i] * (dis[i] - t))
        ac.st(ans)
        return

    @staticmethod
    def lg_p5663(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5663
        tag: shortest_odd_path|shortest_even_path
        """
        n, m, q = ac.read_list_ints()
        inf = math.inf
        graph = WeightedGraphForDijkstra(n, inf)
        for i in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)

        dis = graph.bfs_for_shortest_path_with_odd_and_even(0)
        for _ in range(q):
            x, y = ac.read_list_ints()
            x -= 1
            if dis[x * 2 + y % 2] > y:
                ac.no()
            else:
                ac.yes()
        return

    @staticmethod
    def lg_p5683(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5683
        tag: several_dijkstra|shortest_path|brute_force
        """
        n, m = ac.read_list_ints()
        inf = m + 1
        graph = WeightedGraphForDijkstra(n, inf)
        nums = []
        while len(nums) < 2 * m:
            nums.extend(ac.read_list_ints_minus_one())
        for i in range(0, 2 * m, 2):
            x, y = nums[i], nums[i + 1]
            graph.add_undirected_edge(x, y, 1)
        s1, t1, s2, t2 = ac.read_list_ints()
        s1 -= 1
        s2 -= 1
        dis0 = graph.dijkstra_for_shortest_path(0)
        dis1 = graph.dijkstra_for_shortest_path(s1)
        dis2 = graph.dijkstra_for_shortest_path(s2)
        ans = inf
        for i in range(n):
            cur = dis0[i] + dis1[i] + dis2[i]
            if dis1[i] + dis0[i] <= t1 and dis2[i] + dis0[i] <= t2:
                ans = min(ans, cur)
        ac.st(-1 if ans == inf else m - ans)
        return

    @staticmethod
    def lg_p5837(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5837
        tag: dijkstra|several_params
        """
        # Dijkstra变形问题，带多个状态
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, c, f = ac.read_list_ints()
            dct[i - 1].append([j - 1, c, f])
            dct[j - 1].append([i - 1, c, f])

        dis = [math.inf] * n
        stack = [[0, 0, 0, math.inf]]
        dis[0] = 0
        while stack:
            d, i, cost, flow = heappop(stack)
            if dis[i] < d:
                continue
            for j, c, f in dct[i]:
                dj = -min(flow, f) / (cost + c)
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj, j, cost + c, min(flow, f)])
        ac.st(int(-dis[-1] * 10 ** 6))
        return

    @staticmethod
    def lg_p5930(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5930
        tag: dijkstra|minimum_max_weight_on_shortest_path
        """
        # 接雨水 Dijkstra 
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        visit = [[math.inf] * n for _ in range(n)]
        stack = []
        for i in [0, m - 1]:
            for j in range(n):
                stack.append([grid[i][j], i, j])
                visit[i][j] = grid[i][j]
        for j in [0, n - 1]:
            for i in range(1, m - 1):
                stack.append([grid[i][j], i, j])
                visit[i][j] = grid[i][j]
        heapify(stack)
        while stack:
            d, i, j = heappop(stack)
            if visit[i][j] < d:
                continue
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n:
                    # 每条路径边权最大值当中的最小值
                    dj = max(grid[x][y], d)
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heappush(stack, [dj, x, y])
        ans = 0
        for i in range(m):
            for j in range(n):
                ans += visit[i][j] - grid[i][j]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6063(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6063
        tag: dijkstra|minimum_max_weight_on_shortest_path
        """
        # Dijkstra应用接雨水
        n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        # 虚拟化超级汇点初始化起点
        stack = []
        for i in [0, m - 1]:
            for j in range(n):
                stack.append([grid[i][j], i, j])
        for i in range(1, m - 1):
            for j in [0, n - 1]:
                stack.append([grid[i][j], i, j])
        heapify(stack)

        # shortest_path算法寻找每个格子到达超级汇点的路径途中最大值里面的最小值
        ans = 0
        while stack:
            dis, i, j = heappop(stack)
            if grid[i][j] == -1:
                continue
            ans += 0 if dis < grid[i][j] else dis - grid[i][j]
            grid[i][j] = -1
            for x, y in [[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] != -1:
                    heappush(stack, [dis if dis > grid[x][y] else grid[x][y], x, y])
        ac.st(ans)
        return

    @staticmethod
    def lc_2714_1(n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-shortest-path-with-k-hops/
        tag: limited_shortest_path|layered_dijkstra
        """
        # limited_shortest_path，也可以分层 Dijkstra 求解
        dct = [[] for _ in range(n)]
        for u, v, w in edges:
            dct[u].append([v, w])
            dct[v].append([u, w])

        visit = [[math.inf] * (k + 1) for _ in range(n)]
        stack = [[0, 0, s]]
        visit[s][0] = 0
        while stack:
            dis, c, i = heappop(stack)
            if i == d:
                return dis
            if visit[i][c] < dis:
                continue
            for j, w in dct[i]:
                if c + 1 <= k and dis < visit[j][c + 1]:
                    visit[j][c + 1] = dis
                    heappush(stack, [dis, c + 1, j])
                if dis + w < visit[j][c]:
                    visit[j][c] = dis + w
                    heappush(stack, [dis + w, c, j])
        return -1

    @staticmethod
    def lc_2714_2(n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-shortest-path-with-k-hops/
        tag: limited_shortest_path|layered_dijkstra
        """
        # limited_shortest_path，也可以分层 Dijkstra 求解
        dct = [[] for _ in range(n)]
        for u, v, w in edges:
            dct[u].append([v, w])
            dct[v].append([u, w])

        n = len(dct)
        cnt = [math.inf] * n
        stack = [[0, 0, s]]
        while stack:
            dis, c, i = heappop(stack)
            if i == d:
                return dis
            if cnt[i] < c:
                continue
            cnt[i] = c
            for j, w in dct[i]:
                if c + 1 < cnt[j] and c + 1 <= k:
                    heappush(stack, [dis, c + 1, j])
                if c < cnt[j]:
                    heappush(stack, [dis + w, c, j])
        return -1

    @staticmethod
    def lc_2577(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/
        tag: dijkstra|matrix
        """
        # Dijkstra变形二维矩阵题目

        m, n = len(grid), len(grid[0])
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1

        dis = [[math.inf] * n for _ in range(m)]
        dis[0][0] = 0
        stack = [[0, 0, 0]]

        while stack:
            d, i, j = heappop(stack)
            if dis[i][j] < d:
                continue
            for x, y in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n:
                    if grid[x][y] <= d + 1:
                        dj = d + 1
                    else:
                        dj = d + 1 + 2 * ((grid[x][y] - d - 1 + 1) // 2)
                    if dj < dis[x][y]:
                        dis[x][y] = dj
                        heappush(stack, [dj, x, y])
        return dis[-1][-1]

    @staticmethod
    def lc_2699(n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/modify-graph-edge-weights/
        tag: dijkstra|shortest_path|greedy|brain_teaser
        """
        dct = [[] for _ in range(n)]
        m = len(edges)
        book = [0] * m
        for ind, (i, j, w) in enumerate(edges):
            if w == -1:
                w = 1
                book[ind] = 1
                edges[ind][-1] = w
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍shortest_path最小情况下的距离
        dis0 = [math.inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            return []

        # 第二遍shortest_path
        dis1 = [math.inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是shortest_path上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）|上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def lg_p6512(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6512
        tag: shortest_path
        """
        n, m, k = ac.read_list_ints()
        inf = 10 ** 18
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        dis = [graph.dijkstra_for_shortest_path(i) for i in range(n)]

        dp = [-inf] * (k + 1)
        dp[0] = 0
        pos = [[0, 0]] + [ac.read_list_ints() for _ in range(k)]
        pos.sort()
        for i in range(k):
            t, v = pos[i + 1]
            v -= 1
            lst = [dp[j] + 1 for j in range(i + 1) if dis[v][pos[j][1] - 1] + pos[j][0] <= t] + [0]
            dp[i + 1] = max(lst)
        ac.st(max(dp))
        return

    @staticmethod
    def lg_p8385(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8385
        tag: brain_teaser|build_graph|shortest_path
        """
        # brain_teaserbuild_graph|shortest_path
        n = ac.read_int()
        price = [ac.read_int() for _ in range(n)]
        dct = [[] for _ in range(2 * n)]
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dct[a].append([b, c])
            dct[a + n].append([b + n, c])
        for i in range(n):
            dct[i].append([i + n, price[i] / 2])

        dis = [math.inf] * 2 * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            total, i = heappop(stack)
            if dis[i] < total:
                continue
            for j, w in dct[i]:
                dj = total + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        ac.st(int(dis[n]))
        return

    @staticmethod
    def lc_1786(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-restricted-paths-from-first-to-last-node/
        tag: dijkstra|limited_shortest_path|counter|dag|undirected_to_dag
        """
        # dijkstralimited_shortest_pathcounter（类似shortest_pathcounter）
        dct = defaultdict(dict)
        for i, j, w in edges:
            dct[i - 1][j - 1] = w
            dct[j - 1][i - 1] = w
        mod = 10 ** 9 + 7
        # reverse_order|shortest_path搜寻
        dis = [float('math.inf')] * n
        cnt = [0] * n
        cnt[n - 1] = 1
        dis[n - 1] = 0
        # 定义好初始值
        stack = [[0, n - 1]]
        while stack:
            cur_dis, cur = heappop(stack)
            if dis[cur] < cur_dis:
                continue
            dis[cur] = cur_dis
            for nex in dct[cur]:
                # 如果到达下一个点更近，则更新值
                if dis[nex] > dis[cur] + dct[cur][nex]:
                    dis[nex] = dis[cur] + dct[cur][nex]
                    heappush(stack, [dis[nex], nex])
                # 可以形成有效的路径
                if dis[cur] < dis[nex]:
                    cnt[nex] += cnt[cur]
                    cnt[nex] %= mod
        return cnt[0]

    @staticmethod
    def lc_1928_1(max_time: int, edges: List[List[int]], passing_fees: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/
        tag: dijkstra|limited_shortest_path|floyd
        """
        n = len(passing_fees)
        graph = LimitedWeightedGraph(n)
        for i, j, t in edges:
            graph.add_directed_edge(i, j, t, passing_fees[j])
            graph.add_directed_edge(j, i, t, passing_fees[i])

        ans = graph.limited_dijkstra_tuple(0, n - 1, max_time + 1, passing_fees[0])
        return ans

    @staticmethod
    def lc_1928_2(max_time: int, edges: List[List[int]], passing_fees: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/
        tag: dijkstra|limited_shortest_path|floyd
        """
        # Dijkstralimited_shortest_path，也可根据无后效性类似Floyd的动态规划求解
        n = len(passing_fees)
        dp = [[math.inf] * (max_time + 1) for _ in range(n)]
        dp[0][0] = passing_fees[0]
        for t in range(max_time + 1):
            for i, j, w in edges:
                if w <= t:
                    a, b = dp[j][t], dp[i][t - w] + passing_fees[j]
                    dp[j][t] = a if a < b else b

                    a, b = dp[i][t], dp[j][t - w] + passing_fees[i]
                    dp[i][t] = a if a < b else b
        ans = min(dp[-1])
        return ans if ans < math.inf else -1

    @staticmethod
    def lc_1976(n: int, roads: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/
        tag: dijkstra|number_of_shortest_path|classical
        """
        mod = 10 ** 9 + 7
        inf = 10 ** 15
        graph = WeightedGraphForDijkstra(n, inf)
        for i, j, w in roads:
            graph.add_undirected_edge(i, j, w)
        _, cnt = graph.dijkstra_for_cnt_of_shortest_path(0, 0, mod)
        return cnt[n - 1]

    @staticmethod
    def abc_142f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc142/tasks/abc142_f
        tag: directed|directed_smallest_circle
        """
        n, m = ac.read_list_ints()
        inf = m + 1
        graph = WeightedGraphForDijkstra(n, inf)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        for i, j in edges:
            graph.add_directed_edge(i, j, 1)

        ans = inf
        res = []
        for x, (i, j) in enumerate(edges):
            graph.edge_weight[x + 1] = inf
            path, dis = graph.bfs_for_shortest_path_from_src_to_dst(j, i)
            if dis < ans:
                ans = dis
                res = path[:]
            graph.edge_weight[x + 1] = 1
        if ans == inf:
            ac.st(-1)
            return
        ac.st(len(res))
        for a in res:
            ac.st(a + 1)
        return

    @staticmethod
    def ac_3628(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3631/
        tag: shortest_path_spanning_tree
        """
        # shortest_path_spanning_tree模板题
        n, m, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x].append([y, w, ind])
            dct[y].append([x, w, ind])

        for i in range(n):
            dct[i].sort()

        # 先跑一遍shortest_path
        dis = [math.inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w, _ in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))

        # 选择lexicographical_order较小的边建立shortest_path树
        edge = [[] for _ in range(n)]
        visit = [0] * n
        for i in range(n):
            for j, w, ind in dct[i]:
                if visit[j]:
                    continue
                if dis[i] + w == dis[j]:
                    edge[i].append([j, ind])
                    edge[j].append([i, ind])
                    visit[j] = 1

        # 最后一遍bfs确定选择的边
        ans = []
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            for y, ind in edge[x]:
                if y != fa:
                    ans.append(ind + 1)
                    stack.append([y, x])
        ans = ans[:k]
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def ac_3772(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3775/
        tag: build_graph|reverse_graph|dijkstra|shortest_path|counter|greedy|implemention
        """
        n, m = ac.read_list_ints()
        inf = m + 1
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            graph.add_directed_edge(v, u, 1)

        k = ac.read_int()
        p = ac.read_list_ints_minus_one()
        dis, cnt = graph.bfs_for_cnt_of_shortest_path(p[-1], 0)

        floor = 0
        for i in range(k - 1):
            if k - i - 1 == dis[p[i]]:
                break
            if dis[p[i - 1]] == dis[p[i]] + 1:
                continue
            else:
                floor += 1

        ceil = 0
        for i in range(k - 1):
            if dis[p[i]] == dis[p[i + 1]] + 1 and cnt[p[i]] == cnt[p[i + 1]]:
                continue
            else:
                ceil += 1
        ac.lst([floor, ceil])
        return

    @staticmethod
    def ac_3797(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3800/
        tag: shortest_path|brute_force|sort|greedy
        """
        n, m, k = ac.read_list_ints()
        color = ac.read_list_ints_minus_one()
        inf = m + 1
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)

        dis1 = graph.bfs_for_shortest_path(0)
        dis2 = graph.bfs_for_shortest_path(n - 1)
        color = [(dis1[i] - dis2[i]) * n + i for i in color]
        color.sort()
        ans = -inf
        post = -inf
        for i in range(k - 1, -1, -1):
            ans = max(ans, min(dis1[n - 1], post + dis1[color[i] % n] + 1))
            post = max(post, dis2[color[i] % n])
        ac.st(ans)
        return

    @staticmethod
    def ac_4196(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4199/
        tag: shortest_path
        """
        n, m = ac.read_list_ints()
        inf = 10 ** 15
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        path, ans = graph.dijkstra_for_shortest_path_from_src_to_dst(0, n - 1)
        if ans == inf:
            ac.st(-1)
        else:
            ac.lst([x + 1 for x in path])
        return

    @staticmethod
    def cf_1915g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1915/problem/G
        tag: shortest_path|limited_shortest_path|dijkstra
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            nums = [ac.read_list_ints() for _ in range(m)]
            s = ac.read_list_ints()
            for u, v, w in nums:
                dct[u - 1].append([v - 1, w])
                dct[v - 1].append([u - 1, w])

            n = len(dct)
            stack = [(0, 0, s[0])]
            vis = [math.inf] * n

            while stack:
                d, i, k = heappop(stack)
                if i == n - 1:
                    ac.st(d)
                    break
                if vis[i] <= k:
                    continue
                vis[i] = k
                ss = min(k, s[i])
                for j, w in dct[i]:
                    dj = d + ss * w
                    if ss < vis[j]:
                        heappush(stack, (dj, j, ss))
        return

    @staticmethod
    def abc_342e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc342/tasks/abc342_e
        tag: classical|dijkstra|longest_path
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            ll, dd, k, c, a, b = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[b].append((a, ll, dd, k, c))
        n = len(dct)
        ceil = 5 * 10 ** 18
        dis = [ceil] * n
        dis[n - 1] = -ceil
        stack = [(-ceil, n - 1)]
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, ll, dd, k, c in dct[i]:
                ki = (-d - c - ll) // dd
                if ki >= 0:
                    ki = min(ki, k - 1)
                    dj = -(ll + ki * dd)
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))
        for d in dis[:-1]:
            if d == ceil:
                ac.st("Unreachable")
            else:
                ac.st(-d)
        return

    @staticmethod
    def abc_348d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc348/tasks/abc348_d
        tag: bfs|dijkstra|limited_shortest_path|state|classical
        """
        m, n = ac.read_list_ints()
        visit = [[-1] * n for _ in range(m)]
        grid = [ac.read_str() for _ in range(m)]
        start = [-1, -1]
        end = [-1, -1]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "S":
                    start = [i, j]
                elif grid[i][j] == "T":
                    end = [i, j]
        power = [[0] * n for _ in range(m)]
        for _ in range(ac.read_int()):
            r, c, w = ac.read_list_ints()
            power[r - 1][c - 1] = w
        stack = [(-power[start[0]][start[1]], start[0], start[1])]
        heapify(stack)
        visit[start[0]][start[1]] = power[start[0]][start[1]]
        while stack:
            x, i, j = heappop(stack)
            if x >= 0:
                break
            for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= a < m and 0 <= b < n and grid[a][b] != "#":
                    nex = max(-x - 1, power[a][b])
                    if nex > visit[a][b]:
                        visit[a][b] = nex
                        heappush(stack, (-nex, a, b))
                        if [a, b] == end:
                            ac.yes()
                            return
        ac.st("Yes" if visit[end[0]][end[1]] > -1 else "No")
        return

    @staticmethod
    def abc_257f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc257/tasks/abc257_f
        tag: shortest_path|brute_force|bfs|classical
        """
        n, m = ac.read_list_ints()
        inf = m + 1
        graph = WeightedGraphForDijkstra(n + 1, inf)
        for _ in range(m):
            i, j = ac.read_list_ints()
            graph.add_undirected_edge(i, j, 1)

        dis1 = graph.bfs_for_shortest_path(1)
        dis2 = graph.bfs_for_shortest_path(n)
        ans = []

        for i in range(1, n + 1):
            cur = min(dis1[n], min(dis1[0], dis1[i]) + min(dis2[0], dis2[i]))
            ans.append(cur if cur < inf else -1)
        ac.lst(ans)
        return

    @staticmethod
    def lc_3112(n: int, edges: List[List[int]], disappear: List[int]) -> List[int]:
        """
        url: https://leetcode.com/problems/minimum-time-to-visit-disappearing-nodes/description/
        tag: dijkstra|template|classical
        """
        dct = [[] for _ in range(n)]
        for i, j, t in edges:
            dct[i].append((j, t))
            dct[j].append((i, t))
        initial = 0
        src = 0
        dis = [math.inf] * n
        stack = [(initial, src)]
        dis[src] = initial

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = d + w
                if dj < dis[j] and disappear[j] > dj:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        return [x if x < math.inf else -1 for x in dis]

    @staticmethod
    def abc_252e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc252/tasks/abc252_e
        tag: shortest_path_spanning_tree|dijkstra|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_list_ints_minus_one()
            dct[x].append((y, w + 1, ind))
            dct[y].append((x, w + 1, ind))

        dis = [math.inf] * n
        stack = [(0, 0)]
        dis[0] = 0
        father = [-1] * n
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w, ind in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    father[j] = ind
                    heappush(stack, (dj, j))
        ac.lst([x + 1 for x in father if x != -1])
        return

    @staticmethod
    def abc_245g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc245/tasks/abc245_g
        tag: shortest_path|second_shortest_path|dijkstra|brain_teaser|classical
        """
        n, m, k, ll = ac.read_list_ints()
        a = ac.read_list_ints_minus_one()
        b = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            dct[i].append((j, w + 1))
            dct[j].append((i, w + 1))
        dis1 = [math.inf] * n
        dis2 = [math.inf] * n
        fa1 = [math.inf] * n
        fa2 = [math.inf] * n

        stack = []
        for i in b:
            fa1[i] = a[i]
            dis1[i] = 0
            stack.append((0, i, a[i]))

        while stack:
            d, i, color = heappop(stack)
            if color == fa1[i] and dis1[i] < d:
                continue
            if color == fa2[i] and dis2[i] < d:
                continue
            if dis2[i] < d:
                continue
            for j, w in dct[i]:
                dj = d + w
                if dj < dis1[j]:
                    if dis1[j] < dis2[j] and fa1[j] != color:
                        dis2[j] = dis1[j]
                        fa2[j] = fa1[j]
                    dis1[j] = dj
                    fa1[j] = color
                    heappush(stack, (dj, j, color))
                elif dj < dis2[j] and color != fa1[j]:
                    dis2[j] = dj
                    fa2[j] = color
                    heappush(stack, (dj, j, color))
        ans = [dis2[i] if fa1[i] == a[i] else dis1[i] for i in range(n)]
        ans = [x if x < math.inf else -1 for x in ans]
        ac.lst(ans)
        return

    @staticmethod
    def abc_237e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc237/tasks/abc237_e
        tag: dijkstra|negative_weight|graph_mapping|brain_teaser|classical
        """
        n, m = ac.read_list_ints()
        h = ac.read_list_ints()
        inf = 10 ** 18
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            graph.add_directed_edge(u, v, max(h[v] - h[u], 0))
            graph.add_directed_edge(v, u, max(h[u] - h[v], 0))
        dis = graph.dijkstra_for_shortest_path(0)
        ans = max(h[0] - h[i] - dis[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def cf_449bb(ac=FastIO()):
        """
        url: https://codeforces.com/contest/449/problem/B
        tag: shortest_path|not_shortest_path_spanning_tree|union_find
        """
        n, m, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x].append((y, w))
            dct[y].append((x, w))

        dis = [math.inf] * n
        use = [0] * n
        dis[0] = 0
        for i in range(k):
            s, y = ac.read_list_ints()
            s -= 1
            if y < dis[s]:
                use[s] = 1
                dis[s] = y

        stack = [dis[i] * n + i for i in range(n) if dis[i] < math.inf] + [0]
        heapify(stack)
        while stack:
            x = heappop(stack)
            d, i = x // n, x % n
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    use[j] = 0
                    heappush(stack, dj * n + j)
                elif dj == dis[j]:
                    use[j] = 0
        ac.st(k - sum(use))
        return

    @staticmethod
    def cf_1307d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1307/D
        tag: math|graph|shortest_path|observation|brute_force
        """
        n, m, k = ac.read_list_ints()
        color = ac.read_list_ints_minus_one()
        inf = 10 ** 6
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)

        dis1 = graph.bfs_for_shortest_path(0)
        dis2 = graph.bfs_for_shortest_path(n - 1)
        color = [(dis1[i] - dis2[i]) * n + i for i in color]
        color.sort()
        ans = -inf
        post = -inf
        for i in range(k - 1, -1, -1):
            ans = max(ans, min(dis1[n - 1], post + dis1[color[i] % n] + 1))
            post = max(post, dis2[color[i] % n])
        ac.st(ans)
        return

    @staticmethod
    def cf_938d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/938/D
        tag: dijkstra|fake_source|build_graph|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i].append((j, w))
            dct[j].append((i, w))
        a = ac.read_list_ints()
        for i in range(n):
            dct[0].append((i + 1, a[i]))

        n += 1
        dis = [math.inf] * n
        stack = [(0, 0)]

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = d + w if not i else d + w * 2
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        ac.lst(dis[1:])
        return

    @staticmethod
    def cf_1817b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1817/B
        tag: undirected_shortest_circle|brute_force
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            edges = [ac.read_list_ints_minus_one() for _ in range(m)]
            dct = [set() for _ in range(n)]
            degree = [0] * n
            for i, j in edges:
                dct[i].add(j)
                dct[j].add(i)
                degree[i] += 1
                degree[j] += 1

            parent = [-1] * n
            dis = [math.inf] * n
            for x, y in edges:
                dct[x].discard(y)
                dct[y].discard(x)
                for i in range(n):
                    parent[i] = -1
                    dis[i] = math.inf

                stack = [x]
                dis[x] = 0
                while stack:
                    nex = []
                    for i in stack:
                        for j in dct[i]:
                            if dis[j] == math.inf:
                                dis[j] = dis[i] + 1
                                parent[j] = i
                                nex.append(j)
                    stack = nex[:]
                if dis[y] < math.inf:
                    lst = [y]
                    while lst[-1] != x:
                        lst.append(parent[lst[-1]])
                    if any(degree[z] >= 4 for z in lst):
                        ac.st("YES")
                        i = [z for z in lst if degree[z] >= 4][0]
                        k = len(lst)
                        res = k + 2
                        ac.st(res)
                        for j in range(k):
                            ac.lst([lst[j] + 1, lst[(j + 1) % k] + 1])
                        cnt = 0
                        cur = set(lst)
                        for j in dct[i]:
                            if j not in cur:
                                ac.lst([i + 1, j + 1])
                                cnt += 1
                                if cnt == 2:
                                    break
                        break
                dct[x].add(y)
                dct[y].add(x)
            else:
                ac.st("NO")
        return

    @staticmethod
    def cf_1473e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1473/E
        tag: layer_dijkstra|observation|classical|brain_teaser
        """

        n, m = ac.read_list_ints()  # TLE
        dct = [[] for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            dct[u - 1].append((v - 1, w))
            dct[v - 1].append((u - 1, w))

        n = len(dct)
        dis = [math.inf] * n * 4
        ans = [math.inf] * n
        stack = [(0, 0)]
        dis[0] = ans[0] = 0

        while stack:
            d, s = heappop(stack)
            if dis[s] < d:
                continue
            i = s >> 2
            s1 = s & 1
            s2 = s & 2
            if d < ans[i] and s1 and s2:
                ans[i] = d
            for j, w in dct[i]:
                dj = d + w
                if dj < dis[(j << 2) | s1 | s2]:
                    dis[(j << 2) | s1 | s2] = dj
                    heappush(stack, (dj, (j << 2) | s1 | s2))

                if not s1:
                    dj = d
                    if dj < dis[(j << 2) | 1 | s2]:
                        dis[(j << 2) | 1 | s2] = dj
                        heappush(stack, (dj, (j << 2) | 1 | s2))
                    if not s2:
                        dj = d + w
                        if dj < dis[(j << 2) | 1 | 2]:
                            dis[(j << 2) | 1 | 2] = dj
                            heappush(stack, (dj, (j << 2) | 1 | 2))

                if not s2:
                    dj = d + w * 2
                    if dj < dis[(j << 2) | s1 | 2]:
                        dis[(j << 2) | s1 | 2] = dj
                        heappush(stack, (dj, (j << 2) | s1 | 2))

        ac.lst(ans[1:])
        return

    @staticmethod
    def lg_p5764(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5764
        tag: several_dijkstra|shortest_path
        """
        n, m = ac.read_list_ints()
        lst = [0] + ac.read_list_ints_minus_one()
        inf = 10 ** 18
        graph = WeightedGraphForDijkstra(n, inf)
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        dis = [graph.dijkstra_for_shortest_path(x, 0) for x in lst]
        ans = inf
        for item in permutations(range(1, 6), 5):
            cur = [0] + list(item)
            cost = sum(dis[cur[i]][lst[cur[i + 1]]] for i in range(5))
            ans = min(ans, cost)
        ac.st(ans)
        return

    @staticmethod
    def abc_204e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc204/tasks/abc204_e
        tag: dijkstra|shortest_path|classical|observation
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c, d = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a].append((b, c, d))
            dct[b].append((a, c, d))

        n = len(dct)
        dis = [math.inf] * n
        stack = [(0, 0)]
        dis[0] = 0

        while stack:
            pre, i = heappop(stack)
            if dis[i] < pre:
                continue
            for j, c, d in dct[i]:
                t = max(pre, int(d ** 0.5))
                dd = t + c + d // (t + 1)
                if dd < dis[j]:
                    dis[j] = dd
                    heappush(stack, (dd, j))
        ac.st(dis[-1] if dis[-1] < math.inf else -1)
        return

    @staticmethod
    def cf_545e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/545/E
        tag: shortest_path_spanning_tree|minimum_weight|dijkstra|classical|greedy|shortest_path_mst
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        weight = []
        for ind in range(m):
            x, y, w = ac.read_list_ints_minus_one()
            dct[x].append((y, ind))
            dct[y].append((x, ind))
            weight.append(w + 1)

        root = ac.read_int() - 1
        dis = [math.inf] * n
        stack = [root]
        dis[root] = 0
        father = [-1] * n
        weights = [math.inf] * n
        while stack:
            val = heappop(stack)
            d, i = val // n, val % n
            if dis[i] < d:
                continue
            for j, ind in dct[i]:
                w = weight[ind]
                dj = w + d
                if dj < dis[j] or (dj == dis[j] and w < weights[j]):
                    dis[j] = dj
                    father[j] = ind
                    weights[j] = w
                    heappush(stack, dj * n + j)
        res = [x + 1 for x in father if x != -1]
        ac.st(sum(weight[x - 1] for x in res))
        ac.lst(res)
        return

    @staticmethod
    def cf_786b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/786/problem/B
        tag: segment_tree_opt_build_graph|dijkstra|classical|weighted_graph
        """

        n, q, s = ac.read_list_ints()
        s -= 1
        graph = WeightedGraph(4 * n)
        tree = SegmentTreeOptBuildGraphZKW(n)
        for i in range(1, n):
            graph.add_directed_edge(i, i << 1, 0)
            graph.add_directed_edge(i, (i << 1) | 1, 0)
            graph.add_directed_edge((i << 1) + 2 * n, i + 2 * n, 0)
            graph.add_directed_edge(((i << 1) | 1) + 2 * n, i + 2 * n, 0)

        for i in range(n):
            graph.add_directed_edge(i + 3 * n, i + n, 0)
            graph.add_directed_edge(i + n, i + 3 * n, 0)

        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                v, u, w = [x - 1 for x in lst[1:]]
                graph.add_directed_edge(v + n, u + n, w + 1)
            elif lst[0] == 2:
                v, ll, rr, w = [x - 1 for x in lst[1:]]
                lst = tree.range_opt(ll, rr)
                for j in lst:
                    graph.add_directed_edge(v + 3 * n, j, w + 1)
            else:
                v, ll, rr, w = [x - 1 for x in lst[1:]]
                lst = tree.range_opt(ll, rr)
                for j in lst:
                    graph.add_directed_edge(j + 2 * n, v + n, w + 1)

        graph.dijkstra(s + 3 * n)
        for i in range(n, 2 * n):
            if graph.dis[i] == math.inf:
                graph.dis[i] = -1
        ac.lst(graph.dis[n:2 * n])
        return

    @staticmethod
    def cf_1076d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1076/D
        tag: weighted_graph|shortest_path_mst
        """
        n, m, k = ac.read_list_ints()
        graph = WeightedGraphForShortestPathMST(n)
        for i in range(m):
            x, y, w = ac.read_list_ints_minus_one()
            w += 1
            graph.add_undirected_edge(x, y, w)
        res = graph.shortest_path_mst(0, k)
        ac.st(len(res))
        ac.lst(res)
        return

    @staticmethod
    def abc_375g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc375/tasks/abc375_g
        tag: dijkstra_for_cnt_of_shortest_path|key_edge_in_shortest_path
        """
        n, m = ac.read_list_ints()
        graph = WeightedGraphForDijkstra(n, 2 * 10 ** 14)
        for ind in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            w += 1
            graph.add_undirected_edge(i, j, w)
        assert graph.edge_id == 2 * m + 1

        mod = random.getrandbits(32)
        dis1, cnt1 = graph.dijkstra_for_cnt_of_shortest_path(0, 0, mod)
        dis2, cnt2 = graph.dijkstra_for_cnt_of_shortest_path(n - 1, 0, mod)

        for ind in range(m):
            i, j, w = graph.edge_from[ind * 2 + 1], graph.edge_to[ind * 2 + 1], graph.edge_weight[ind * 2 + 1]
            if dis1[i] + w + dis2[j] == dis1[-1] and cnt1[i] * cnt2[j] % mod == cnt1[-1]:
                ac.yes()
                continue
            i, j = j, i
            if dis1[i] + w + dis2[j] == dis1[-1] and cnt1[i] * cnt2[j] % mod == cnt1[-1]:
                ac.yes()
                continue
            ac.no()
        return
