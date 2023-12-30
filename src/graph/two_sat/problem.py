"""

Algorithm：scc|2-sat|largest_circle|smallest_circle
Description：scc|dag|shrink_point|topological_sort|strongly_connected_component
Example：path pass the most points which can be duplicated
2-SAT：giving n sets, each with two elements, in which (a, b) indicating that a and b are contradictory (where a and b belong to different sets), then select an element from each set and determine if a total of n pairwise non-contradictory elements can be selected

====================================LeetCode====================================

=====================================LuoGu======================================
P4782（https://www.luogu.com.cn/problem/P4782）2-sat|scc|classical
P5782（https://www.luogu.com.cn/problem/P5782）2-sat|scc|classical
P4171（https://www.luogu.com.cn/problem/P4171）2-sat|scc|classical

===================================CodeForces===================================
1438C（https://codeforces.com/problemset/problem/1438/C）2-sat|scc|classical


"""

from src.graph.tarjan.template import TarjanCC
from src.utils.fast_io import FastIO


class TwoSAT:
    def __init__(self):
        return

    @staticmethod
    def lg_p4782(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4782
        tag: 2-sat|scc|classical
        """
        n, m = ac.read_list_ints()
        edge = [[] for _ in range(2 * n)]
        for _ in range(m):
            i, a, j, b = ac.read_list_ints()
            i -= 1
            j -= 1
            edge[i * 2 + 1 - a].append(j * 2 + b)
            edge[j * 2 + 1 - b].append(i * 2 + a)

        _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(2 * n, [list(s) for s in edge])
        for s in scc_node_id:
            pre = set()
            for node in s:
                if node // 2 in pre:
                    ac.st("IMPOSSIBLE")
                    return
                pre.add(node // 2)

        ac.st("POSSIBLE")
        ans = [0] * n
        pre = set()
        for s in scc_node_id:
            for node in s:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        ac.lst(ans)
        return

    @staticmethod
    def lg_p5782(ac=FastIO()):
        """
        url:https://www.luogu.com.cn/problem/P5782
        tag: 2-sat|scc|classical
        """
        n, m = ac.read_list_ints()
        edge = [set() for _ in range(4 * n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            edge[a * 2 + 1].add(b * 2)
            edge[b * 2 + 1].add(a * 2)

        for i in range(n):
            a, b = 2 * i, 2 * i + 1
            edge[a * 2 + 1].add(b * 2)
            edge[a * 2].add(b * 2 + 1)
            edge[b * 2 + 1].add(a * 2)
            edge[b * 2].add(a * 2 + 1)

        _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(4 * n, [list(s) for s in edge])
        for s in scc_node_id:
            pre = set()
            for node in s:
                if node // 2 in pre:
                    ac.st("NIE")
                    return
                pre.add(node // 2)

        ans = [0] * 2 * n
        pre = set()
        for s in scc_node_id:
            for node in s:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        for i in range(2 * n):
            if ans[i]:
                ac.st(i + 1)
        return

    @staticmethod
    def lg_p4171(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4171
        tag: 2-sat|scc|classical
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            edge = [set() for _ in range(2 * n)]
            for _ in range(m):
                lst = ac.read_list_strs()
                i = int(lst[0][1:]) - 1
                j = int(lst[1][1:]) - 1
                a = 1 if lst[0][0] == "h" else 0
                b = 1 if lst[1][0] == "h" else 0
                if i * 2 + (1 - a) != j * 2 + b:
                    edge[i * 2 + (1 - a)].add(j * 2 + b)
                if j * 2 + (1 - b) != i * 2 + a:
                    edge[j * 2 + (1 - b)].add(i * 2 + a)

            _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(2 * n, [list(s) for s in edge])

            ans = True
            for s in scc_node_id:
                pre = set()
                for node in s:
                    if node // 2 in pre:
                        ans = False
                    pre.add(node // 2)
            ac.st("GOOD" if ans else "BAD")
        return

    @staticmethod
    def cf_1438c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1438/C
        tag: 2-sat|scc|classical
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_list_ints() for _ in range(m)]
            edge = [set() for _ in range(2 * m * n)]
            for i in range(m):
                for j in range(n):
                    if i + 1 < m:
                        x, y = i * n + j, i * n + n + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i + 1][j] + b:
                                edge[x * 2 + a].add(y * 2 + 1 - b)
                                edge[y * 2 + b].add(x * 2 + 1 - a)
                    if j + 1 < n:
                        x, y = i * n + j, i * n + 1 + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i][j + 1] + b:
                                edge[x * 2 + a].add(y * 2 + 1 - b)
                                edge[y * 2 + b].add(x * 2 + 1 - a)
            for i in range(2 * m * n):
                if i in edge[i]:
                    edge[i].discard(i)
            _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(2 * m * n, [list(s) for s in edge])

            ans = [0] * m * n
            pre = set()
            for sc in scc_node_id:
                for node in sc:
                    i = node // 2
                    if i not in pre:
                        ans[i] = node % 2
                    pre.add(i)

            for x in range(m * n):
                grid[x // n][x % n] += ans[x]
            for g in grid:
                ac.lst(g)
        return
