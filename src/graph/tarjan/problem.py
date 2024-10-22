"""
Algorithm：tarjan|cut_point|cut_edge|point_doubly_connected_component|edge_doubly_connected_component|pdcc|edcc
Description：scc|edcc|pdcc|cur_point|cut_edge|directed_acyclic_graph

====================================LeetCode====================================
1192（https://leetcode.cn/problems/critical-connections-in-a-network/）tarjan|cut_edge
2360（https://leetcode.cn/problems/longest-cycle-in-a-graph/solution/）largest_circle|scc|topological_sort
2204（https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/description/）scc|dag|build_graph|reverse_graph
1568（https://leetcode.cn/problems/minimum-number-of-days-to-disconnect-island/solution/）cut_point|tarjan

=====================================LuoGu======================================
P3387（https://www.luogu.com.cn/problem/P3387）scc
P3388（https://www.luogu.com.cn/problem/P3388）multi_edge|self_loop|cut_point
P8435（https://www.luogu.com.cn/problem/P8435）multi_edge|self_loop|several_circle
P8436（https://www.luogu.com.cn/problem/P8436）multi_edge|self_loop|build_graph|fake_source
P2860（https://www.luogu.com.cn/problem/P2860）edge_doubly_connected_component|scc|tree_centroid
P2863（https://www.luogu.com.cn/problem/P2863）tarjan|scc

P1656（https://www.luogu.com.cn/problem/P1656）cut_edge
P1793（https://www.luogu.com.cn/problem/P1793）cut_point|brute_force|union_find
P2656（https://www.luogu.com.cn/problem/P2656）scc|dag|longest_path
P1726（https://www.luogu.com.cn/problem/P1726）scc
P2002（https://www.luogu.com.cn/problem/P2002）scc|shrink_point
P2341（https://www.luogu.com.cn/problem/P2341）scc|shrink_point
P2835（https://www.luogu.com.cn/problem/P2835）scc|shrink_point
P2863（https://www.luogu.com.cn/problem/P2863）scc
B3609（https://www.luogu.com.cn/problem/B3609）scc
B3610（https://www.luogu.com.cn/problem/B3610）point_doubly_connected_component
P7033（https://www.luogu.com.cn/problem/P7033）scc|dag|tree_dp
P7965（https://www.luogu.com.cn/problem/P7965）scc|dag|tree_dp

===================================CodeForces===================================
1811F（https://codeforces.com/contest/1811/problem/F）scc|pdcc
427C（https://codeforces.com/problemset/problem/427/C）scc|shrink_point
193A（https://codeforces.com/contest/193/problem/A）brain_teaser|cut_point
999E（https://codeforces.com/contest/999/problem/E）scc|shrink_point|union_find
1213F（https://codeforces.com/contest/1213/problem/F）scc|shrink_point|topological_sort|greedy
1547G（https://codeforces.com/contest/1547/problem/G）scc|shrink_point|build_graph|counter|number_of_path
1702E（https://codeforces.com/contest/1702/problem/E）point_doubly_connected_component|pdcc|undirected|odd_circle
1768D（https://codeforces.com/contest/1768/problem/D）permutation_circle|tarjan
1986E（https://codeforces.com/contest/1986/problem/E）cutting_edge|tarjan|brute_force
118E（https://codeforces.com/problemset/problem/118/E）tarjan|cutting_edge|order|classical
1000E（https://codeforces.com/problemset/problem/1000/E）tarjan|edcc|cutting_edge|tree_diameter

===================================CodeForces===================================
ABC334G（https://atcoder.jp/contests/abc334/tasks/abc334_g）union_find|mod_reverse|tarjan|edcc|expectation|math|classical
ABC334E（https://atcoder.jp/contests/abc334/tasks/abc334_e）union_find|mod_reverse|expectation|math|classical
ABC245F（https://atcoder.jp/contests/abc245/tasks/abc245_f）scc|reverse_graph|implemention|bfs|classical
ABC357E（https://atcoder.jp/contests/abc357/tasks/abc357_e）scc|build_graph|reverse_graph

=====================================AcWing=====================================
3582（https://www.acwing.com/problem/content/3582/）scc
3816（https://www.acwing.com/problem/content/description/3816/）scc|topological_sort|dag_dp

===================================LibraryChecker===================================
1 Cycle Detection (Directed)（https://judge.yosupo.jp/problem/cycle_detection）directed_graph|circle
2 Strongly Connected Components（https://judge.yosupo.jp/problem/scc）scc
3 Two-Edge-Connected Components（https://judge.yosupo.jp/problem/two_edge_connected_components）edcc

"""
import math
from collections import Counter
from typing import List

from src.graph.tarjan.template import Tarjan, DirectedGraphForTarjanScc
from src.graph.topological_sort.template import WeightedGraphForTopologicalSort
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3387(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3387
        tag: scc|dag|classical|longest_path
        """
        n, m = ac.read_list_ints()
        weights = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(x, y)
        graph.build_scc()
        scc_weights = [0] * graph.scc_id
        for i in range(n):
            scc_weights[graph.node_scc_id[i]] += weights[i]

        class Graph(WeightedGraphForTopologicalSort):
            def dag_dp(self):
                dp = [0] * self.n
                stack = [u for u in range(self.n) if not self.degree[u]]
                for u in stack:
                    dp[u] = scc_weights[u]
                while stack:
                    nex = []
                    for u in stack:
                        for v in self.get_to_nodes(u):
                            self.degree[v] -= 1
                            dp[v] = max(dp[v], dp[u] + scc_weights[v])
                            if not self.degree[v]:
                                nex.append(v)
                    stack = nex
                return dp

        graph_topo = Graph(graph.scc_id)
        for val in graph.get_scc_edge_degree()[0]:
            x, y = val // graph.scc_id, val % graph.scc_id
            graph_topo.add_directed_edge(x, y, 1)
        ans = graph_topo.dag_dp()
        ac.st(max(ans))
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/cycle_detection
        tag: directed_graph|circle
        """
        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        dct = dict()
        for ind in range(m):
            i, j = ac.read_list_ints()
            graph.add_directed_original_edge(i, j)
            dct[i * n + j] = ind
        graph.build_scc()

        root = [0] * graph.scc_id
        cnt = [0] * graph.scc_id
        for i in range(n):
            root[graph.node_scc_id[i]] = i
            cnt[graph.node_scc_id[i]] += 1
        visit = [0] * n
        for g in range(graph.scc_id):
            if cnt[g] > 1:
                i = root[g]
                lst = [i]
                visit[lst[-1]] = 1
                end = -1
                while end == -1:
                    ind = graph.point_head[lst[-1]]
                    while ind:
                        j = graph.edge_to[ind]
                        if graph.node_scc_id[j] == g:
                            if visit[j]:
                                end = j
                                break
                            lst.append(j)
                            visit[j] = 1
                            break
                        ind = graph.edge_next[ind]
                ind = lst.index(end)
                lst = lst[ind:] + [end]
                ans = []
                k = len(lst)
                for j in range(1, k):
                    x, y = lst[j - 1], lst[j]
                    ans.append(dct[x * n + y])
                ac.st(len(ans))
                for a in ans:
                    ac.st(a)
                break
        else:
            ac.st(-1)
        return

    @staticmethod
    def library_check_2(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/scc
        tag: scc
        """
        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            x, y = ac.read_list_ints()
            graph.add_directed_original_edge(x, y)
        graph.build_scc()

        graph.initialize_graph()
        for i in range(n):
            graph.add_directed_edge(graph.node_scc_id[i], i)
        ac.st(graph.scc_id)
        for i in range(graph.scc_id - 1, -1, -1):
            lst = []
            ind = graph.point_head[i]
            while ind:
                lst.append(graph.edge_to[ind])
                ind = graph.edge_next[ind]
            k = len(lst)
            ac.lst([k] + lst)
        return

    @staticmethod
    def library_check3(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/two_edge_connected_components
        tag: edcc
        """
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints()
            if a > b:
                a, b = b, a
            if a == b:
                x = len(edge)
                edge.append(set())
                edge[a].add(x)
                edge[x].add(a)
            elif b in edge[a]:
                x = len(edge)
                edge.append(set())
                edge[a].add(x)
                edge[x].add(a)
                edge[b].add(x)
                edge[x].add(b)
            else:
                edge[a].add(b)
                edge[b].add(a)
        group = Tarjan().get_edcc(len(edge), edge)
        res = []
        for r in group:
            lst = [str(x) for x in r if x < n]
            if lst:
                res.append([str(len(lst))] + lst)
        ac.st(len(res))
        for a in res:
            ac.st(" ".join(a))
        return

    @staticmethod
    def lg_p3388(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3388
        tag: multi_edge|self_loop|cut_point
        """
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, _ = Tarjan().get_cut(n, [list(d) for d in dct])
        cut_node = sorted(list(cut_node))
        ac.st(len(cut_node))
        ac.lst([x + 1 for x in cut_node])
        return

    @staticmethod
    def lg_p8435(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8435
        tag: multi_edge|self_loop|several_circle|pdcc
        """
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            if x != y:
                edge[x].add(y)
                edge[y].add(x)
                degree[x] += 1
                degree[y] += 1

        pdcc_id, pdcc_node_id, node_pdcc_id = Tarjan().get_pdcc(n, [list(e) for e in edge])
        ac.st(len(pdcc_node_id) + sum(degree[i] == 0 for i in range(n)))
        for r in range(pdcc_id):
            ac.lst([len(pdcc_node_id[r])] + [x + 1 for x in pdcc_node_id[r]])
        for i in range(n):
            if not degree[i]:
                ac.lst([1, i + 1])
        return

    @staticmethod
    def lg_p8436(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8436
        tag: multi_edge|self_loop|build_graph|fake_source|edcc
        """
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        dup = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if a > b:
                a, b = b, a
            if a == b:
                continue
            if b in edge[a]:
                if b not in dup[a]:
                    dup[a].add(b)
                    x = len(edge)
                    edge.append(set())
                    edge[a].add(x)
                    edge[x].add(a)
                    edge[b].add(x)
                    edge[x].add(b)
            else:
                edge[a].add(b)
                edge[b].add(a)
        group = Tarjan().get_edcc(len(edge), edge)
        res = []
        for r in group:
            lst = [str(x + 1) for x in r if x < n]
            if lst:
                res.append([str(len(lst))] + lst)
        ac.st(len(res))
        for a in res:
            ac.st(" ".join(a))
        return

    @staticmethod
    def cf_999e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/999/problem/E
        tag: scc|shrink_point
        """
        n, m, s = ac.read_list_ints()
        s -= 1
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(i, j)
        graph.build_scc()
        scc_degree = graph.get_scc_edge_degree()[1]
        ans = sum(x == 0 for x in scc_degree)
        ans -= int(scc_degree[graph.node_scc_id[s]] == 0)
        ac.st(ans)
        return

    @staticmethod
    def cf_1702e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1702/problem/E
        tag: point_doubly_connected_component|pdcc|undirected|odd_circle
        """
        for _ in range(ac.read_int()):
            def check():
                n = ac.read_int()
                nums = [ac.read_list_ints() for _ in range(n)]
                cnt = Counter()
                for a, b in nums:
                    if a == b:
                        ac.no()
                        return
                    cnt[a] += 1
                    cnt[b] += 1
                if max(cnt.values()) > 2:
                    ac.no()
                    return

                dct = [[] for _ in range(n)]
                for a, b in nums:
                    a -= 1
                    b -= 1
                    dct[a].append(b)
                    dct[b].append(a)
                group_id, group_node, node_group_id = Tarjan().get_pdcc(n, dct)
                for g in group_node:
                    if len(group_node[g]) % 2:
                        ac.no()
                        return
                ac.yes()
                return

            check()

        return

    @staticmethod
    def lc_1192(n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/critical-connections-in-a-network/
        tag: tarjan|cut_edge
        """
        edge = [set() for _ in range(n)]
        for i, j in connections:
            edge[i].add(j)
            edge[j].add(i)
        cutting_point, cutting_edge = Tarjan().get_cut(n, [list(e) for e in edge])
        return [list(e) for e in cutting_edge]

    @staticmethod
    def lg_p1656(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1656
        tag: cut_edge
        """
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, cut_edge = Tarjan().get_cut(n, [list(d) for d in dct])
        cut_edge = sorted([list(e) for e in cut_edge])
        for x in cut_edge:
            ac.lst([w + 1 for w in x])
        return

    @staticmethod
    def lg_p2860(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2860
        tag: edge_doubly_connected_component|scc|tree_centroid
        """
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        dup = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if a > b:
                a, b = b, a
            if a == b:  # self loop is not necessary
                continue
            if b in edge[a]:
                if b not in dup[a]:  # at most one another duplicate edge
                    dup[a].add(b)
                    dup[b].add(a)
                    x = len(edge)
                    edge.append(set())
                    edge[a].add(x)
                    edge[x].add(a)
                    edge[b].add(x)
                    edge[x].add(b)
            else:
                edge[a].add(b)
                edge[b].add(a)
        group = Tarjan().get_edcc(len(edge), [ls.copy() for ls in edge])
        scc_node_id = []
        for r in group:
            lst = [x for x in r if x < n]
            if lst:
                scc_node_id.append(lst)
        scc_id = len(scc_node_id)
        node_scc_id = [-1] * n
        for i in range(scc_id):
            for j in scc_node_id[i]:
                node_scc_id[j] = i

        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in edge[i]:
                if j < n:
                    a, b = node_scc_id[i], node_scc_id[j]
                    if a != b:
                        new_dct[a].add(b)
                        new_dct[b].add(a)
        leaf = sum(len(ls) == 1 for ls in new_dct)
        ac.st((leaf + 1) // 2)
        return

    @staticmethod
    def lg_p2863(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2863
        tag: scc
        """
        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(i, j)
        graph.build_scc()
        cnt = [0] * graph.scc_id
        for i in range(n):
            cnt[graph.node_scc_id[i]] += 1
        ans = sum(x > 1 for x in cnt)
        ac.st(ans)
        return

    @staticmethod
    def cf_427c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/427/C
        tag: scc|shrink_point
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(ac.read_int()):
            i, j = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(i, j)
        graph.build_scc()
        graph.build_new_graph_from_scc_id_to_original_node()
        ans = 1
        cost = 0
        mod = 10 ** 9 + 7
        for g in range(graph.scc_id):
            cnt = Counter([nums[i] for i in graph.get_original_out_node(g)])
            x = min(cnt)
            cost += x
            ans *= cnt[x]
            ans %= mod
        ac.lst([cost, ans])
        return

    @staticmethod
    def lg_p2656(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2656
        tag: scc|dag|longest_path|longest_path|dijkstra|spfa|reverse_graph
        """

        def check(cc, dd):
            xx = 0
            while cc:
                xx += cc
                cc = cc * dd // 10
            return xx

        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        edges = []
        for _ in range(m):
            a, b, c, d = ac.read_list_strs()
            a = int(a) - 1
            b = int(b) - 1
            c = int(c)
            d = int(float(d) * 10)
            edges.append((a, b, c, d))
            graph.add_directed_original_edge(a, b)
        graph.build_scc()
        scc_weight = [0] * graph.scc_id
        for a, b, c, d in edges:
            if graph.node_scc_id[a] == graph.node_scc_id[b]:
                scc_weight[graph.node_scc_id[a]] += check(c, d)
        graph.initialize_graph()

        # reverse_graph
        s = ac.read_int() - 1
        graph_topo = WeightedGraphForTopologicalSort(graph.scc_id)
        for a, b, c, d in edges:
            if graph.node_scc_id[a] != graph.node_scc_id[b]:
                graph_topo.add_directed_edge(graph.node_scc_id[b], graph.node_scc_id[a], c)
        res = graph_topo.topological_sort_for_dag_dp_with_edge_weight(scc_weight)[graph.node_scc_id[s]]
        ac.st(res)
        return

    @staticmethod
    def lg_p1726(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1726
        tag: scc
        """
        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            a, b, t = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(a, b)
            if t == 1:
                graph.add_directed_original_edge(b, a)
        graph.build_scc()
        scc_node_id = graph.get_scc_node_id()
        ans = []
        for g in scc_node_id:
            lst = sorted(g)
            if len(lst) > len(ans) or (len(lst) == len(ans) and ans > lst):
                ans = lst[:]
        ac.st(len(ans))
        ac.lst([x + 1 for x in ans])
        return

    @staticmethod
    def lg_p2002(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2002
        tag: scc|shrink_point
        """
        n, m = ac.read_list_ints()
        inf = m + 1
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if a >= 0:  # invalid data
                graph.add_directed_original_edge(a, b)
        graph.build_scc()
        _, scc_degree = graph.get_scc_edge_degree()
        ans = sum(x == 0 for x in scc_degree)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2341(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2341
        tag: scc|shrink_point
        """
        n, m = ac.read_list_ints()
        graph = DirectedGraphForTarjanScc(n)
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(a, b)
        graph.build_scc()
        _, scc_degree, scc_cnt = graph.get_scc_edge_degree_reverse()
        ans = sum(x == 0 for x in scc_degree)
        ac.st(0 if ans > 1 else scc_cnt[scc_degree.index(0)])
        return

    @staticmethod
    def lg_p2835(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2835
        tag: scc|shrink_point
        """
        n = ac.read_int()
        graph = DirectedGraphForTarjanScc(n)
        for i in range(n):
            for j in ac.read_list_ints_minus_one()[:-1]:
                graph.add_directed_original_edge(i, j)
        graph.build_scc()
        _, scc_degree = graph.get_scc_edge_degree()
        ac.st(sum(d == 0 for d in scc_degree))
        return

    @staticmethod
    def lg_p7033(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7033
        tag: scc|dag|tree_dp|reverse_graph
        """

        class Graph(DirectedGraphForTarjanScc):
            def get_scc_dag_dp(self):
                scc_edge = set()
                scc_cnt = [0] * self.scc_id
                for i in range(self.n):
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        a, b = self.node_scc_id[i], self.node_scc_id[j]
                        if a != b:
                            scc_edge.add(b * self.scc_id + a)
                        ind = self.edge_next[ind]
                    scc_cnt[self.node_scc_id[i]] += 1
                self.initialize_graph()
                for val in scc_edge:
                    self.add_directed_edge(val // self.scc_id, val % self.scc_id)
                scc_degree = [0] * self.scc_id
                for val in scc_edge:
                    scc_degree[val % self.scc_id] += 1
                ans_group = [0] * graph.scc_id
                stack = [i for i in range(graph.scc_id) if not scc_degree[i]]
                while stack:
                    i = stack.pop()
                    ans_group[i] += scc_cnt[i] - 1
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        ans_group[j] += ans_group[i] + 1
                        stack.append(j)
                        ind = self.edge_next[ind]
                ans = [ans_group[graph.node_scc_id[i]] for i in range(self.n)]
                return ans

        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        graph = Graph(n)
        lst = [nums[i][0] * n + i for i in range(n)]
        lst.sort(reverse=True)
        for i in range(1, n):
            graph.add_directed_original_edge(lst[i - 1] % n, lst[i] % n)
        lst = [nums[i][1] * n + i for i in range(n)]
        lst.sort(reverse=True)
        for i in range(1, n):
            graph.add_directed_original_edge(lst[i - 1] % n, lst[i] % n)

        graph.build_scc()
        final = graph.get_scc_dag_dp()
        for x in final:
            ac.st(x)
        return

    @staticmethod
    def lg_p7965(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7965
        tag: scc|dag|tree_dp
        """

        class Graph(DirectedGraphForTarjanScc):
            def get_scc_dag_dp(self):
                scc_edge = set()
                scc_cnt = [0] * self.scc_id
                for i in range(self.n):
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        a, b = self.node_scc_id[i], self.node_scc_id[j]
                        if a != b:
                            scc_edge.add(b * self.scc_id + a)
                        ind = self.edge_next[ind]
                    scc_cnt[self.node_scc_id[i]] += 1
                self.initialize_graph()
                for val in scc_edge:
                    self.add_directed_edge(val // self.scc_id, val % self.scc_id)
                scc_degree = [0] * self.scc_id
                for val in scc_edge:
                    scc_degree[val % self.scc_id] += 1

                stack = [i for i in range(graph.scc_id) if not scc_degree[i]]
                while stack:
                    i = stack.pop()
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        stack.append(j)
                        depth[j] = depth[i] + 1
                        ancestor[j] = ancestor[i]
                return

        n, m, q = ac.read_list_ints()
        graph = Graph(n)
        for _ in range(m):
            lst = ac.read_list_ints_minus_one()
            for x in range(n):
                graph.add_directed_original_edge(x, lst[x])
        graph.build_scc()
        ancestor = list(range(n))
        depth = [0] * n
        graph.get_scc_dag_dp()
        for _ in range(q):
            u, v = ac.read_list_ints_minus_one()
            x, y = graph.node_scc_id[u], graph.node_scc_id[v]
            ac.st("DA" if ancestor[x] == ancestor[y] and depth[x] <= depth[y] else "NE")
        return

    @staticmethod
    def lc_1568(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-days-to-disconnect-island/solution/
        tag: cut_point|tarjan
        """
        m, n = len(grid), len(grid[0])

        edge = [[] for _ in range(m * n)]
        nodes = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if i + 1 < m and grid[i + 1][j] == 1:
                        edge[i * n + j].append(i * n + n + j)
                        edge[i * n + n + j].append(i * n + j)
                    if j + 1 < n and grid[i][j + 1] == 1:
                        edge[i * n + j].append(i * n + 1 + j)
                        edge[i * n + 1 + j].append(i * n + j)
                    nodes.add(i * n + j)
        if len(nodes) <= 1:
            return len(nodes)
        nodes = sorted(list(nodes))
        ind = {num: i for i, num in enumerate(nodes)}
        k = len(nodes)
        dct = [[] for _ in range(k)]
        for i in range(m * n):
            for j in edge[i]:
                dct[ind[i]].append(ind[j])
                dct[ind[j]].append(ind[i])

        uf = UnionFind(k)
        for i in range(k):
            for j in dct[i]:
                uf.union(i, j)
        if uf.part > 1:
            return 0

        cutting_point, _ = Tarjan().get_cut(k, dct)
        return 2 if not cutting_point else 1

    @staticmethod
    def ac_3582(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3582/
        tag: scc
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            p = ac.read_list_ints_minus_one()
            graph = DirectedGraphForTarjanScc(n)
            for i in range(n):
                graph.add_directed_original_edge(i, p[i])

            graph.build_scc()
            scc_cnt = graph.get_scc_cnt()
            ans = [scc_cnt[graph.node_scc_id[i]] for i in range(n)]
            ac.lst(ans)
        return

    @staticmethod
    def ac_3816(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3816/
        tag: scc|topological_sort|dag_dp
        """

        class Graph(DirectedGraphForTarjanScc):
            def get_scc_dag_dp(self):
                scc_edge = set()
                scc_weight = [0] * self.scc_id * 26
                for i in range(self.n):
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        a, b = self.node_scc_id[i], self.node_scc_id[j]
                        if a != b:
                            scc_edge.add(a * self.scc_id + b)
                        ind = self.edge_next[ind]
                    scc_weight[self.node_scc_id[i] * 26 + lst[i]] += 1
                self.initialize_graph()
                for val in scc_edge:
                    self.add_directed_edge(val // self.scc_id, val % self.scc_id)
                scc_degree = [0] * self.scc_id
                for val in scc_edge:
                    scc_degree[val % self.scc_id] += 1
                res = max(scc_weight)
                stack = [i for i in range(self.scc_id) if not scc_degree[i]]
                scc_pre = [0] * self.scc_id * 26
                while stack:
                    nex = []
                    for i in stack:
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            scc_degree[j] -= 1
                            for w in range(26):
                                scc_pre[j * 26 + w] = max(scc_pre[i * 26 + w] + scc_weight[i * 26 + w],
                                                          scc_pre[j * 26 + w])
                                res = max(res, scc_pre[j * 26 + w] + scc_weight[j * 26 + w])
                            if not scc_degree[j]:
                                nex.append(j)
                            ind = self.edge_next[ind]
                    stack = nex
                return res

        n, m = ac.read_list_ints()
        graph = Graph(n)
        lst = [ord(w) - ord("a") for w in ac.read_str()]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            if u == v:
                ac.st(-1)
                return
            graph.add_directed_original_edge(u, v)
        graph.build_scc()
        if graph.scc_id != n:
            ac.st(-1)
            return
        ans = graph.get_scc_dag_dp()
        ac.st(ans)
        return

    @staticmethod
    def lc_2360(edges):
        """
        url: https://leetcode.cn/problems/longest-cycle-in-a-graph/
        tag: largest_circle|scc|topological_sort|scc
        """
        n = len(edges)
        graph = DirectedGraphForTarjanScc(n)
        for i in range(n):
            if edges[i] != -1:
                graph.add_directed_edge(i, edges[i])
        graph.build_scc()
        ans = max(graph.get_scc_cnt())
        return ans if ans > 1 else -1

    @staticmethod
    def abc_334g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc334/tasks/abc334_g
        tag: union_find|mod_reverse|tarjan|edcc|expectation|math|classical
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        uf = UnionFind(m * n)
        mod = 998244353
        green = 0
        dct = [[] for _ in range(m * n)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "#":
                    if i + 1 < m and grid[i + 1][j] == "#":
                        uf.union(i * n + j, i * n + n + j)
                        dct[i * n + j].append(i * n + n + j)
                        dct[i * n + n + j].append(i * n + j)
                    if j + 1 < n and grid[i][j + 1] == "#":
                        uf.union(i * n + j, i * n + j + 1)
                        dct[i * n + j].append(i * n + 1 + j)
                        dct[i * n + 1 + j].append(i * n + j)
                    green += 1
        group = uf.get_root_part()
        ans = 0
        p = pow(green, -1, mod)
        roots = [root for root in group if grid[root // n][root % n] == "#"]
        count = len(roots)
        _, _, node_group_id = Tarjan().get_pdcc(m * n, dct)
        for root in roots:
            lst = group[root]
            tot = len(lst)
            if tot == 1:
                ans += (count - 1)
            else:
                for ls in lst:
                    ans += count + len(node_group_id[ls]) - 1
                    ans %= mod
        ans *= p
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_245f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc245/tasks/abc245_f
        tag: scc|reverse_graph|implemention|bfs|classical
        """

        class Graph(DirectedGraphForTarjanScc):
            def get_scc_dag_dp(self):
                scc_edge = set()
                scc_cnt = [0] * self.scc_id
                for i in range(self.n):
                    ind = self.point_head[i]
                    while ind:
                        j = self.edge_to[ind]
                        a, b = self.node_scc_id[i], self.node_scc_id[j]
                        if a != b:
                            scc_edge.add(b * self.scc_id + a)
                        ind = self.edge_next[ind]
                    scc_cnt[self.node_scc_id[i]] += 1
                self.initialize_graph()
                for val in scc_edge:
                    self.add_directed_edge(val // self.scc_id, val % self.scc_id)
                scc_degree = [0] * self.scc_id
                for val in scc_edge:
                    scc_degree[val % self.scc_id] += 1
                stack = [i for i in range(self.scc_id) if scc_cnt[i] > 1]
                visit = [0] * self.scc_id
                for i in stack:
                    visit[i] = scc_cnt[i]
                while stack:
                    nex = []
                    for i in stack:
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if not visit[j]:
                                nex.append(j)
                                visit[j] = scc_cnt[j]
                            ind = self.edge_next[ind]
                    stack = nex
                return sum(visit)

        n, m = ac.read_list_ints()
        graph = Graph(n)
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            graph.add_directed_original_edge(u, v)
        graph.build_scc()
        ans = graph.get_scc_dag_dp()
        ac.st(ans)
        return

    @staticmethod
    def abc_357e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc357/tasks/abc357_e
        tag: scc|build_graph|reverse_graph
        """
        n = ac.read_int()
        nums = ac.read_list_ints_minus_one()
        graph = DirectedGraphForTarjanScc(n)
        for i in range(n):
            graph.add_directed_original_edge(i, nums[i])
        graph.build_scc()
        ans = graph.get_scc_dag_dp()
        ac.st(ans)
        return

    @staticmethod
    def cf_1986e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1986/problem/E
        tag: cutting_edge|tarjan|brute_force
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)
            _, edge = Tarjan().get_cut(n, dct)
            if not edge:
                ac.st(n * (n - 1) // 2)
            else:
                uf = UnionFind(n)
                new_dct = [[] for _ in range(n)]
                for i, j in edge:
                    uf.union(i, j)
                    new_dct[i].append(j)
                    new_dct[j].append(i)
                for i in range(n):
                    for j in dct[i]:
                        if uf.union(i, j):
                            new_dct[i].append(j)
                            new_dct[j].append(i)
                sub = [0] * n
                stack = [(0, -1)]
                father = [-1] * n
                while stack:
                    i, fa = stack.pop()
                    if i >= 0:
                        stack.append((~i, fa))
                        for j in new_dct[i]:
                            if j != fa:
                                stack.append((j, i))
                                father[j] = i
                    else:
                        i = ~i
                        sub[i] = 1
                        for j in new_dct[i]:
                            if j != fa:
                                sub[i] += sub[j]
                ans = n * (n - 1) // 2
                for i, j in edge:
                    if father[i] == j:
                        cur = sub[i]
                    else:
                        cur = sub[j]
                    ans = min(ans, cur * (cur - 1) // 2 + (n - cur) * (n - cur - 1) // 2)
                ac.st(ans)
        return

    @staticmethod
    def cf_118e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/118/E
        tag: tarjan|cutting_edge|order|classical
        """
        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        nums = []
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)
            nums.append((x, y))

        res = Tarjan().get_cut(n, edge)
        cutting_point, cutting_edge, parent, order = res
        if not cutting_edge:
            for x, y in nums:
                if x == parent[y]:
                    ac.lst([x + 1, y + 1])
                elif y == parent[x]:
                    ac.lst([y + 1, x + 1])
                elif order[x] > order[y]:
                    ac.lst([x + 1, y + 1])
                else:
                    ac.lst([y + 1, x + 1])
        else:
            ac.st(0)
        return

    @staticmethod
    def cf_1000e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1000/E
        tag: tarjan|edcc|cutting_edge|tree_diameter
        """
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        _, cutting_edges = Tarjan().get_cut(n, [list(e) for e in dct])
        # edcc
        nex_dct = [list(e) for e in dct]
        for i, j in cutting_edges:
            dct[i].discard(j)
            dct[j].discard(i)
        visit = [0] * n
        node_edcc_id = [-1] * n
        edcc_id = 0
        for i in range(n):
            if visit[i]:
                continue
            stack = [i]
            visit[i] = 1
            node_edcc_id[i] = edcc_id
            while stack:
                x = stack.pop()
                for j in dct[x]:
                    if not visit[j]:
                        visit[j] = 1
                        stack.append(j)
                        node_edcc_id[j] = edcc_id
            edcc_id += 1
        del dct, visit
        new_dct = [[] for _ in range(edcc_id)]
        for i in range(n):
            for j in nex_dct[i]:
                a, b = node_edcc_id[i], node_edcc_id[j]
                if a != b:
                    new_dct[a].append(b)
        del nex_dct
        n = len(new_dct)
        root = 0
        dis = [math.inf] * n
        stack = [(root, -1)]
        dis[root] = 0
        while stack:
            i, fa = stack.pop()
            for j in new_dct[i]:  # weighted edge
                if j != fa:
                    dis[j] = dis[i] + 1
                    stack.append((j, i))
        root = dis.index(max(dis))
        dis = [math.inf] * n
        stack = [(root, -1)]
        dis[root] = 0
        while stack:
            i, fa = stack.pop()
            for j in new_dct[i]:  # weighted edge
                if j != fa:
                    dis[j] = dis[i] + 1
                    stack.append((j, i))

        ac.st(max(dis))
        return
