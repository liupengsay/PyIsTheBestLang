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

===================================CodeForces===================================
ABC334G（https://atcoder.jp/contests/abc334/tasks/abc334_g）union_find|mod_reverse|tarjan|edcc|expectation|math|classical
ABC334E（https://atcoder.jp/contests/abc334/tasks/abc334_e）union_find|mod_reverse|expectation|math|classical
ABC245F（https://atcoder.jp/contests/abc245/tasks/abc245_f）scc|reverse_graph|implemention|bfs|classical

=====================================AcWing=====================================
3582（https://www.acwing.com/problem/content/3582/）scc
3816（https://www.acwing.com/problem/content/description/3816/）scc|topological_sort|dag_dp

===================================LibraryChecker===================================
1 Cycle Detection (Directed)（https://judge.yosupo.jp/problem/cycle_detection）directed_graph|circle
2 Strongly Connected Components（https://judge.yosupo.jp/problem/scc）scc
3 Two-Edge-Connected Components（https://judge.yosupo.jp/problem/two_edge_connected_components）edcc

"""
from collections import Counter
from collections import deque
from typing import List

from src.graph.dijkstra.template import Dijkstra
from src.graph.tarjan.template import Tarjan
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
        weight = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edge[x].add(y)
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, [list(e) for e in edge])

        new_dct = [set() for _ in range(scc_id)]
        new_weight = [sum(weight[j] for j in scc_node_id[i]) for i in range(scc_id)]
        for i in range(n):
            for j in edge[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[a].add(b)
        new_degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1

        ans = [0] * scc_id
        stack = deque([i for i in range(scc_id) if not new_degree[i]])
        for i in stack:
            ans[i] = new_weight[i]
        while stack:
            i = stack.popleft()
            for j in new_dct[i]:
                w = new_weight[j]
                new_degree[j] -= 1
                if ans[i] + w > ans[j]:
                    ans[j] = ans[i] + w
                if not new_degree[j]:
                    stack.append(j)
        ac.st(max(ans))
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/cycle_detection
        tag: directed_graph|circle
        """
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for i in range(m):
            u, v = ac.read_list_ints()
            dct[u][v] = i

        edge = [list(d.keys()) for d in dct]
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, edge)
        for g in scc_node_id:
            if len(scc_node_id[g]) > 1:
                nodes_set = scc_node_id[g]
                i = list(scc_node_id[g])[0]
                lst = [i]
                pre = {i}
                end = -1
                while end == -1:
                    for j in edge[i]:
                        if j in nodes_set:
                            if j in pre:
                                end = j
                                break
                            lst.append(j)
                            i = j
                            pre.add(j)
                            break
                ind = lst.index(end)
                lst = lst[ind:] + [end]
                ans = []
                k = len(lst)
                for j in range(1, k):
                    x, y = lst[j - 1], lst[j]
                    ans.append(dct[x][y])
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
        dct = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints()
            if x != y:
                dct[x].add(y)
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, [list(e) for e in dct])
        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in dct[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[a].add(b)
        new_degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        ans = []
        stack = [i for i in range(scc_id) if not new_degree[i]]
        while stack:
            ans.extend(stack)
            nex = []
            for i in stack:
                for j in new_dct[i]:
                    new_degree[j] -= 1
                    if not new_degree[j]:
                        nex.append(j)
            stack = nex
        ac.st(len(ans))
        for i in ans:
            k = len(scc_node_id[i])
            ac.lst([k] + list(scc_node_id[i]))
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
        edges = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edges[x].add(y)
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, [list(e) for e in edges])

        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in edges[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[a].add(b)
        new_degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        ans = sum(x == 0 for x in new_degree)
        ans -= int(new_degree[node_scc_id[s]] == 0)
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
                        ac.st("NO")
                        return
                    cnt[a] += 1
                    cnt[b] += 1
                if max(cnt.values()) > 2:
                    ac.st("NO")
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
                        ac.st("NO")
                        return
                ac.st("YES")
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
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            edge[a].add(b)
        _, group, _ = Tarjan().get_scc(n, [list(e) for e in edge])
        ac.st(sum(len(g) > 1 for g in group))
        return

    @staticmethod
    def cf_427c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/427/C
        tag: scc|shrink_point
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
        _, group, _ = Tarjan().get_scc(n, dct)
        ans = 1
        cost = 0
        mod = 10 ** 9 + 7
        for g in group:
            cnt = Counter([nums[i] for i in group[g]])
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
        tag: scc|dag|longest_path|longest_path|dijkstra|spfa
        """

        def check(cc, dd):
            xx = 0
            while cc:
                xx += cc
                cc = cc * dd // 10
            return xx

        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        edges = []
        for _ in range(m):
            a, b, c, d = ac.read_list_strs()
            a = int(a) - 1
            b = int(b) - 1
            c = int(c)
            d = int(float(d) * 10)
            edges.append((a, b, c, d))
            if a != b:
                dct[a].add(b)

        s = ac.read_int() - 1

        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, [list(x) for x in dct])

        cnt = [0] * scc_id
        for a, b, c, d in edges:
            if node_scc_id[a] == node_scc_id[b]:
                cnt[node_scc_id[a]] += check(c, d)

        new_dct = [[] for _ in range(scc_id)]
        for a, b, c, d in edges:
            if node_scc_id[a] != node_scc_id[b]:
                new_dct[node_scc_id[a]].append((node_scc_id[b], c + cnt[node_scc_id[b]]))
        dis = Dijkstra().get_longest_path(new_dct, node_scc_id[s], cnt[node_scc_id[s]])
        ac.st(max(dis))
        return

    @staticmethod
    def lg_p1726(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1726
        tag: scc
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, t = ac.read_list_ints_minus_one()
            dct[a].append(b)
            if t == 1:
                dct[b].append(a)
        _, scc_node_id, _ = Tarjan().get_scc(n, dct)
        ans = []
        for g in scc_node_id:
            lst = sorted(list(scc_node_id[g]))
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
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if i != j:
                dct[i].add(j)
        scc_id, _, node_scc_id = Tarjan().get_scc(n, [list(e) for e in dct])

        in_degree = [0] * scc_id
        for i in range(n):
            a = node_scc_id[i]
            for j in dct[i]:
                b = node_scc_id[j]
                if a != b:
                    in_degree[b] = 1
        ac.st(sum(x == 0 for x in in_degree))
        return

    @staticmethod
    def lg_p2341(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2341
        tag: scc|shrink_point
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints()
            if a != b:
                dct[a - 1].append(b - 1)
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, dct)

        degree = [0] * scc_id
        for i in range(n):
            for j in dct[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    degree[a] += 1
        degree = [i for i in range(scc_id) if not degree[i]]
        if len(degree) > 1:
            ac.st(0)
        else:
            ac.st(len(scc_node_id[degree[0]]))
        return

    @staticmethod
    def lg_p2835(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2835
        tag: scc|shrink_point
        """
        n = ac.read_int()
        edge = [ac.read_list_ints_minus_one()[:-1] for _ in range(n)]
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, edge)
        degree = [0] * scc_id
        for i in range(n):
            for j in edge[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    degree[b] = 1
        ac.st(sum(d == 0 for d in degree))
        return

    @staticmethod
    def lg_p7033(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7033
        tag: scc|dag|tree_dp|reverse_graph
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ind = list(range(n))
        dct = [[] for _ in range(n)]
        degree = [0] * n

        ind.sort(key=lambda it: -nums[it][0])
        for i in range(n - 1):
            x, y = ind[i], ind[i + 1]
            degree[y] += 1
            dct[x].append(y)

        ind.sort(key=lambda it: -nums[it][1])
        for i in range(n - 1):
            x, y = ind[i], ind[i + 1]
            degree[y] += 1
            dct[x].append(y)
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, [list(set(e)) for e in dct])
        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in dct[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[b].add(a)
        new_degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1

        ans_group = [0] * scc_id
        stack = [i for i in range(scc_id) if not new_degree[i]]
        while stack:
            i = stack.pop()
            ans_group[i] += len(scc_node_id[i]) - 1
            for j in new_dct[i]:
                ans_group[j] += ans_group[i] + 1
                stack.append(j)
        for i in range(n):
            ac.st(ans_group[node_scc_id[i]])
        return

    @staticmethod
    def lg_p7965(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7965
        tag: scc|dag|tree_dp
        """
        n, m, q = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            lst = ac.read_list_ints_minus_one()
            for i in range(n):
                if lst[i] != i:
                    dct[i].add(lst[i])
        dct = [list(e) for e in dct]
        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, dct)

        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in dct[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[a].add(b)
        degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                degree[j] += 1
        depth = [0] * n
        ancestor = list(range(n))
        stack = [i for i in range(scc_id) if not degree[i]]
        while stack:
            i = stack.pop()
            for j in new_dct[i]:
                stack.append(j)
                depth[j] = depth[i] + 1
                ancestor[j] = ancestor[i]

        for _ in range(q):
            a, b = ac.read_list_ints_minus_one()
            x, y = node_scc_id[a], node_scc_id[b]
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
            p = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for i in range(n):
                dct[i].append(p[i] - 1)
            _, group, _ = Tarjan().get_scc(n, dct)
            ans = [0] * n
            for g in group:
                x = len(group[g])
                for i in group[g]:
                    ans[i] = x
            ac.lst(ans)
        return

    @staticmethod
    def ac_3816(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3816/
        tag: scc|topological_sort|dag_dp
        """
        n, m = ac.read_list_ints()
        s = ac.read_str()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if a == b:
                ac.st(-1)
                return
            dct[a].add(b)
        scc_id, _, _ = Tarjan().get_scc(n, [list(e) for e in dct])
        if scc_id != n:
            ac.st(-1)
            return

        cur = [[0] * 26 for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            for j in dct[i]:
                degree[j] += 1
        stack = [i for i in range(n) if not degree[i]]
        ans = 0
        while stack:
            nex = []
            for i in stack:
                cur[i][ord(s[i]) - ord("a")] += 1
                x = max(cur[i])
                if x > ans:
                    ans = x
                for j in dct[i]:
                    for w in range(26):
                        y = cur[i][w]
                        if y > cur[j][w]:
                            cur[j][w] = y
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex[:]
        ac.st(ans)
        return

    @staticmethod
    def lc_2360(edges):
        """
        url: https://leetcode.cn/problems/longest-cycle-in-a-graph/
        tag: largest_circle|scc|topological_sort|scc
        """
        n = len(edges)
        dct = [[] for _ in range(n)]
        for i in range(n):
            if edges[i] != -1:
                dct[i].append(edges[i])

        _, scc_node_id, _ = Tarjan().get_scc(n, dct)
        ans = max(len(ls) for ls in scc_node_id)
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
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        rev = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            degree[j] += 1
            dct[i].append(j)
            rev[j].append(i)

        scc_id, scc_node_id, node_scc_id = Tarjan().get_scc(n, dct)
        stack = []
        for ls in scc_node_id:
            if len(ls) > 1:
                stack.extend(ls)

        visit = [0] * n
        for i in stack:
            visit[i] = 1
        while stack:
            nex = []
            for i in stack:
                for j in rev[i]:
                    if not visit[j]:
                        visit[j] = 1
                        nex.append(j)
            stack = nex
        ac.st(sum(x > 0 for x in visit))
        return