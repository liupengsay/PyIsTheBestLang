"""
Algorithm：tarjan|cut_point|cut_edge|point_doubly_connected_component|edge_doubly_connected_component|pdcc|edcc
Description：scc|edcc|pdcc|cur_point|cut_edge|directed_acyclic_graph

====================================LeetCode====================================
1192（https://leetcode.com/problems/critical-connections-in-a-network/）tarjan|cut_edge
2360（https://leetcode.com/problems/longest-cycle-in-a-graph/solution/）largest_circle|scc|topological_sort
2204（https://leetcode.com/problems/distance-to-a-cycle-in-undirected-graph/description/）scc|dag|build_graph|reverse_graph
1568（https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/solution/）cut_point|tarjan

=====================================LuoGu======================================
3388（https://www.luogu.com.cn/problem/P3388）multi_edge|self_loop|cut_point
8435（https://www.luogu.com.cn/problem/P8435）multi_edge|self_loop|several_circle
8436（https://www.luogu.com.cn/problem/P8436）multi_edge|self_loop|build_graph|fake_source
2860（https://www.luogu.com.cn/problem/P2860）edge_doubly_connected_component|scc|tree_centroid
2863（https://www.luogu.com.cn/problem/P2863）tarjan|scc

1656（https://www.luogu.com.cn/problem/P1656）cut_edge
1793（https://www.luogu.com.cn/problem/P1793）cut_point|brute_force|union_find
2656（https://www.luogu.com.cn/problem/P2656）scc|dag|longest_path
1726（https://www.luogu.com.cn/problem/P1726）scc
2002（https://www.luogu.com.cn/problem/P2002）scc|shrink_point
2341（https://www.luogu.com.cn/problem/P2341）scc|shrink_point
2835（https://www.luogu.com.cn/problem/P2835）scc|shrink_point
2863（https://www.luogu.com.cn/problem/P2863）scc
3609（https://www.luogu.com.cn/problem/B3609）scc
3610（https://www.luogu.com.cn/problem/B3610）point_doubly_connected_component
7033（https://www.luogu.com.cn/problem/P7033）scc|dag|tree_dp
7965（https://www.luogu.com.cn/problem/P7965）scc|dag|tree_dp

===================================CodeForces===================================
1811F（https://codeforces.com/contest/1811/problem/F）scc|pdcc
427C（https://codeforces.com/problemset/problem/427/C）scc|shrink_point
193A（https://codeforces.com/contest/193/problem/A）brain_teaser|cut_point
999E（https://codeforces.com/contest/999/problem/E）scc|shrink_point
1213F（https://codeforces.com/contest/1213/problem/F）scc|shrink_point|topological_sort|greedy
1547G（https://codeforces.com/contest/1547/problem/G）scc|shrink_point|build_graph|counter|number_of_path
1702E（https://codeforces.com/contest/1702/problem/E）point_doubly_connected_component|pdcc|undirected|odd_circle
1768D（https://codeforces.com/contest/1768/problem/D）permutation_circle|tarjan

=====================================AcWing=====================================
3579（https://www.acwing.com/problem/content/3582/）scc
3813（https://www.acwing.com/problem/content/submission/3816/）scc|topological_sort|dag_dp

===================================LibraryChecker===================================
1 Cycle Detection (Directed)（https://judge.yosupo.jp/problem/cycle_detection）directed_graph|circle
2 Strongly Connected Components（https://judge.yosupo.jp/problem/scc）scc
3 Two-Edge-Connected Components（https://judge.yosupo.jp/problem/two_edge_connected_components）edcc

"""
import copy
from collections import Counter
from collections import defaultdict, deque
from math import inf
from typing import List

from src.dp.tree_dp.template import ReRootDP
from src.graph.tarjan.template import TarjanCC, TarjanUndirected, TarjanDirected
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for i in range(m):
            u, v = ac.read_list_ints()
            dct[u][v] = i

        edge = [list(d.keys()) for d in dct]
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, edge)
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
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints()
            if x != y:
                dct[x].add(y)
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in dct])
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
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints()
            # 需要处理自环与重边
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
        group = TarjanCC().get_edge_doubly_connected_component_bfs(len(edge), edge)
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
    def lc_2360_1(edges: List[int]) -> int:
        # TarjanCC 求 scc 有向图scc
        n = len(edges)
        edge = [set() for _ in range(n)]
        for i in range(n):
            if edges[i] != -1:
                edge[i].add(edges[i])
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in edge])
        ans = max(len(scc_node_id[r]) for r in scc_node_id)
        return ans if ans > 1 else -1

    @staticmethod
    def lc_2360_2(edges: List[int]) -> int:
        # 有向图 Tarjan 求 scc 有向图scc
        n = len(edges)
        edge = [[] for _ in range(n)]
        for i in range(n):
            if edges[i] != -1:
                edge[i] = [edges[i]]
        _, _, sub_group = TarjanDirected().check_graph(edge, n)
        ans = -1
        for sub in sub_group:
            if len(sub) > 1 and len(sub) > ans:
                ans = len(sub)
        return ans

    @staticmethod
    def lg_p3388(ac=FastIO()):
        # 模板: TarjanCC 求无向图cut_point
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, _ = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(d) for d in dct])
        cut_node = sorted(list(cut_node))
        ac.st(len(cut_node))
        ac.lst([x + 1 for x in cut_node])
        return

    @staticmethod
    def lg_p8435(ac=FastIO()):
        # 模板: TarjanCC 求无向图point_doubly_connected_component连通分量
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            # 注意自环的特殊处理
            if x != y:
                edge[x].add(y)
                edge[y].add(x)
                degree[x] += 1
                degree[y] += 1

        pdcc_id, pdcc_node_id, node_pdcc_id = TarjanCC().get_point_doubly_connected_component_bfs(n, [list(e) for e in
                                                                                                      edge])
        ac.st(len(pdcc_node_id) + sum(degree[i] == 0 for i in range(n)))
        for r in pdcc_node_id:
            ac.lst([len(pdcc_node_id[r])] + [x + 1 for x in pdcc_node_id[r]])
        for i in range(n):
            if not degree[i]:
                ac.lst([1, i + 1])
        return

    @staticmethod
    def lg_p8436(ac=FastIO()):
        # 模板: TarjanCC 求无向图edge_doubly_connected_component连通分量
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            # 需要处理自环与重边
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
        group = TarjanCC().get_edge_doubly_connected_component_bfs(len(edge), edge)
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
        # scc|shrink_point|后查看入度为0的点个数
        n, m, s = ac.read_list_ints()
        s -= 1
        edges = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edges[x].add(y)
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in edges])
        # 建立新图
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
        # point_doubly_connected_component无向图找环，判断有无奇数环
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
                group_id, group_node, node_group_id = TarjanCC().get_point_doubly_connected_component_bfs(n, dct)
                for g in group_node:
                    if len(group_node[g]) % 2:
                        ac.st("NO")
                        return

                ac.st("YES")
                return

            check()

        return

    @staticmethod
    def lc_1192_1(n: int, connections: List[List[int]]) -> List[List[int]]:
        #  TarjanCC 求cut_edge
        edge = [set() for _ in range(n)]
        for i, j in connections:
            edge[i].add(j)
            edge[j].add(i)
        cutting_point, cutting_edge = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        return [list(e) for e in cutting_edge]

    @staticmethod
    def lc_1192_2(n: int, connections: List[List[int]]) -> List[List[int]]:
        #  Tarjan 求cut_edge
        edge = [[] for _ in range(n)]
        for i, j in connections:
            edge[i].append(j)
            edge[j].append(i)
        cut_edge, cut_node, sub_group = TarjanUndirected().check_graph(edge, n)
        return cut_edge

    @staticmethod
    def lg_p1656(ac=FastIO()):
        # tarjan求无向图cut_edge
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, cut_edge = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(d) for d in dct])
        cut_edge = sorted([list(e) for e in cut_edge])
        for x in cut_edge:
            ac.lst([w + 1 for w in x])
        return

    @staticmethod
    def lg_p2860(ac=FastIO()):
        # 模板: TarjanCC 求无向图edge_doubly_connected_component连通分量缩点后，质心为根时的叶子数
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        degree = defaultdict(int)
        pre = set()
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            # 需要处理自环与重边
            if a > b:
                a, b = b, a
            if a == b:
                x = len(edge)
                edge.append(set())
                edge[a].add(x)
                degree[a] += 1
                degree[x] += 1
                edge[x].add(a)
            elif (a, b) in pre:
                x = len(edge)
                edge.append(set())
                edge[a].add(x)
                edge[x].add(a)
                edge[b].add(x)
                edge[x].add(b)
                degree[a] += 1
                degree[x] += 2
                degree[b] += 1
            else:
                pre.add((a, b))
                edge[a].add(b)
                edge[b].add(a)
                degree[a] += 1
                degree[b] += 1
        group = TarjanCC().get_edge_doubly_connected_component_bfs(len(edge), copy.deepcopy(edge))

        # 建立新图
        res = []
        for r in group:
            lst = [x for x in r if x < n]
            if lst:
                res.append(lst)

        k = len(res)
        if k == 1:
            ac.st(0)
            return
        ind = dict()
        for i in range(k):
            for x in res[i]:
                ind[x] = i
        dct = [set() for _ in range(k)]
        for i in range(len(edge)):
            for j in edge[i]:
                if i < n and j < n and ind[i] != ind[j]:
                    dct[ind[i]].add(ind[j])
                    dct[ind[j]].add(ind[i])
        dct = [list(s) for s in dct]

        # 求树的质心
        center = ReRootDP().get_tree_centroid(dct)
        stack = [[center, -1]]
        ans = 0
        while stack:
            i, fa = stack.pop()
            cnt = 0
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    cnt += 1
            if not cnt:
                ans += 1
        ac.st((ans + 1) // 2)
        return

    @staticmethod
    def lg_p2863(ac=FastIO()):
        # 模板: TarjanCC 求scc
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            edge[a].add(b)
        _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in edge])

        ans = 0
        for g in group:
            ans += len(group[g]) > 1
        ac.st(ans)
        return

    @staticmethod
    def cf_427c(ac=FastIO()):
        # tarjan有向图缩点后counter
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
        _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
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
        # scc缩点后，DAG最长路
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        edges = []
        for _ in range(m):
            a, b, c, d = ac.read_list_strs()
            a = int(a) - 1
            b = int(b) - 1
            c = int(c)
            d = float(d)
            edges.append([a, b, c, d])
            if a != b:
                edge[a].add(b)
        s = ac.read_int() - 1
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(x) for x in edge])
        cnt = defaultdict(int)
        dis = [defaultdict(int) for _ in range(scc_id)]
        pre = [set() for _ in range(scc_id)]
        for i, j, c, d in edges:
            if node_scc_id[i] == node_scc_id[j]:
                x = 0
                while c:
                    x += c
                    c = int(c * d)
                cnt[node_scc_id[i]] += x
            else:
                a, b = node_scc_id[i], node_scc_id[j]
                dis[a][b] = ac.max(dis[a][b], c)
                pre[b].add(a)

        # 注意这里可能有 0 之外的入度为 0 的点，需要先拓扑消除
        stack = deque([i for i in range(scc_id) if not pre[i] and i != node_scc_id[s]])
        while stack:
            i = stack.popleft()
            for j in dis[i]:
                pre[j].discard(i)
                if not pre[j]:
                    stack.append(j)

        # 广搜最长路，进一步还可以确定相应的具体路径
        visit = [-inf] * scc_id
        visit[node_scc_id[s]] = cnt[node_scc_id[s]]
        stack = deque([node_scc_id[s]])
        while stack:
            i = stack.popleft()
            for j in dis[i]:
                w = dis[i][j]
                pre[j].discard(i)
                if visit[i] + w + cnt[j] > visit[j]:
                    visit[j] = visit[i] + w + cnt[j]
                if not pre[j]:
                    stack.append(j)
        ac.st(max(visit))
        return

    @staticmethod
    def lg_p1726(ac=FastIO()):
        # scc裸题
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, t = ac.read_list_ints_minus_one()
            dct[a].append(b)
            if t == 1:
                # 双向通行
                dct[b].append(a)
        _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
        # 获得新图
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
        # scc缩点后，入度为0的节点个数
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if i != j:
                dct[i].add(j)
        # 必须要缩点，否则单独一个环没办法获取消息
        scc_id, _, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in dct])

        # 新图
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
        # scc缩点后出度为 0 的点集个数与大小
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints()
            if a != b:
                dct[a - 1].append(b - 1)
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, dct)

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
        # sccscc缩点后入度为 0 的点数
        n = ac.read_int()
        edge = [ac.read_list_ints_minus_one()[:-1] for _ in range(n)]
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, edge)
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
        # scc缩点后 DAG tree_dp|
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
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n,
                                                                                           [list(set(e)) for e in dct])

        # 建立新图
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

        # 结果
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
        # scc缩点后 DAG tree_dp|
        n, m, q = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            lst = ac.read_list_ints()
            for i in range(n):
                dct[i].add(lst[i] - 1)
        dct = [list(e) for e in dct]
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, dct)

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
        sub = [set() for _ in range(scc_id)]
        stack = [i for i in range(scc_id) if not degree[i]]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in new_dct[i]:
                    stack.append(j)
            else:
                i = ~i
                sub[i].add(i)
                for j in new_dct[i]:
                    for x in sub[j]:
                        sub[i].add(x)
        for _ in range(q):
            a, b = ac.read_list_ints_minus_one()
            x, y = node_scc_id[a], node_scc_id[b]
            ac.st("DA" if y in sub[x] else "NE")
        return

    @staticmethod
    def lc_1568(grid: List[List[int]]) -> int:
        # 求连通分量与cut_point数量题
        m, n = len(grid), len(grid[0])

        # build_graph|
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
        # 特殊情况
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

        cutting_point, _ = TarjanCC().get_cutting_point_and_cutting_edge_bfs(k, dct)
        return 2 if not cutting_point else 1

    @staticmethod
    def ac_3549(ac=FastIO()):
        # scc模板题
        for _ in range(ac.read_int()):
            n = ac.read_int()
            p = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for i in range(n):
                dct[i].append(p[i] - 1)
            _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
            ans = [0] * n
            for g in group:
                x = len(group[g])
                for i in group[g]:
                    ans[i] = x
            ac.lst(ans)
        return

    @staticmethod
    def ac_3813(ac=FastIO()):
        # scc模板与topological_sortingDP
        n, m = ac.read_list_ints()
        s = ac.read_str()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if a == b:
                ac.st(-1)
                return
            dct[a].add(b)
        scc_id, _, _ = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in dct])
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