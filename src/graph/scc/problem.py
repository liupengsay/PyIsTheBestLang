"""

Algorithm：scc|2-sat|largest_circle|smallest_circle
Description：scc|dag|shrink_point|topological_sort|strongly_connected_component
Example：path pass the most points which can be duplicated
2-SAT：giving n sets, each with two elements, in which (a, b) indicating that a and b are contradictory (where a and b belong to different sets), then select an element from each set and determine if a total of n pairwise non-contradictory elements can be selected

====================================LeetCode====================================
2360（https://leetcode.com/problems/longest-cycle-in-a-graph/）largest_circle|scc|topological_sort|scc

=====================================LuoGu======================================
P3387（https://www.luogu.com.cn/problem/P3387）scc
P2661（https://www.luogu.com.cn/problem/P2661）smallest_circle|directed_circle_based_tree|topological_sort
P4089（https://www.luogu.com.cn/problem/P4089）circle|scc|self_loop
P5145（https://www.luogu.com.cn/problem/P5145）circle_based_tree|scc

P4782（https://www.luogu.com.cn/problem/P4782）2-sat|scc|classical
P5782（https://www.luogu.com.cn/problem/P5782）2-sat|scc|classical
P4171（https://www.luogu.com.cn/problem/P4171）2-sat|scc|classical
===================================CodeForces===================================
1438C（https://codeforces.com/problemset/problem/1438/C）2-sat|scc|classical


"""

from collections import deque

from src.graph.scc.template import Tarjan, Kosaraju
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3387(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3387
        tag: scc
        """
        # 有向图scc将环缩点后求最长路
        n, m = ac.read_list_ints()
        weight = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edge[x].add(y)
        edge = [list(e) for e in edge]

        # 求得scc后重新build_graph|，这里也可以 Kosaraju 算法
        tarjan = Tarjan(edge)
        ind = [-1] * n
        m = len(tarjan.scc)
        point = [0] * m
        degree = [0] * m
        dct = [[] for _ in range(n)]
        for i, ls in enumerate(tarjan.scc):
            for j in ls:
                ind[j] = i
                point[i] += weight[j]
        for i in range(n):
            for j in edge[i]:
                u, v = ind[i], ind[j]
                if u != v:
                    dct[u].append(v)
        for i in range(m):
            for j in dct[i]:
                degree[j] += 1

        # topological_sorting求最长路，这里也可以dfs|
        visit = [0] * m
        stack = deque([i for i in range(m) if not degree[i]])
        for i in stack:
            visit[i] = point[i]
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                w = point[j]
                degree[j] -= 1
                if visit[i] + w > visit[j]:
                    visit[j] = visit[i] + w
                if not degree[j]:
                    stack.append(j)
        ac.st(max(visit))
        return

    @staticmethod
    def lc_2360(edges):
        """
        url: https://leetcode.com/problems/longest-cycle-in-a-graph/
        tag: largest_circle|scc|topological_sort|scc
        """

        # 模板: 求内向circle_based_tree的最大权值和环 edge表示有向边 i 到 edge[i] 而 dct表示对应的边权值
        def largest_circle(n, edge, dct):

            def dfs(x, sum_):
                nonlocal ans
                if x == st:
                    ans = ans if ans > sum_ else sum_
                    return
                # 访问过
                if a[x] or b[x]:
                    return
                a[x] = 1
                dfs(edge[x], sum_ + dct[x])
                a[x] = 0
                return

            a = [0] * n
            b = [0] * n
            ans = 0
            for st in range(n):
                dfs(edge[st], dct[st])
                b[st] = 1
            return ans

        # 题目也可用 scc 或者topological_sorting求解
        return largest_circle(len(edges), edges, [1] * len(edges))


class TwoSAT:
    def __init__(self):
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
            # build_graph|并把索引编码
            edge = [[] for _ in range(2 * m * n)]
            for i in range(m):
                for j in range(n):
                    if i + 1 < m:
                        x, y = i * n + j, i * n + n + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i + 1][j] + b:
                                edge[x * 2 + a].append(y * 2 + 1 - b)
                                edge[y * 2 + b].append(x * 2 + 1 - a)
                    if j + 1 < n:
                        x, y = i * n + j, i * n + 1 + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i][j + 1] + b:
                                edge[x * 2 + a].append(y * 2 + 1 - b)
                                edge[y * 2 + b].append(x * 2 + 1 - a)

            #####################################################
            # 按照强连通缩点
            tarjan = Tarjan(edge)
            # specific_plan赋予，先出现的确定值
            ans = [0] * m * n
            pre = set()
            for sc in tarjan.scc:
                for node in sc:
                    i = node // 2
                    if i not in pre:
                        ans[i] = node % 2
                    pre.add(i)

            for x in range(m * n):
                grid[x // n][x % n] += ans[x]
            for g in grid:
                ac.lst(g)
            #####################################################
            # Kosaraju算法
            kosaraju = Kosaraju(2 * m * n, edge)
            # 注意是小于符号
            ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(m * n)]
            for x in range(m * n):
                grid[x // n][x % n] += ans[x]
            for g in grid:
                ac.lst(g)
            return grid

    @staticmethod
    def luogu_4782(ac=FastIO()):
        n, m = ac.read_list_ints()
        # build_graph|并把索引编码
        edge = [[] for _ in range(2 * n)]
        for _ in range(m):
            i, a, j, b = ac.read_list_ints()
            i -= 1
            j -= 1
            edge[i * 2 + 1 - a].append(j * 2 + b)
            edge[j * 2 + 1 - b].append(i * 2 + a)

        #####################################################
        # 按照强连通缩点检验是否存在冲突
        tarjan = Tarjan(edge)
        for sc in tarjan.scc:
            pre = set()
            for node in sc:
                # 条件相反的两个点不能在一个scc
                if node // 2 in pre:
                    ac.st("IMPOSSIBLE")
                    return
                pre.add(node // 2)

        # specific_plan赋予，先出现的确定值
        ac.st("POSSIBLE")
        ans = [0] * n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        ac.lst(ans)

        #####################################################
        # Kosaraju算法（与上面的算法二选一）
        kosaraju = Kosaraju(2 * n, edge)
        for i in range(n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                ac.st("IMPOSSIBLE")
                return

        ac.st("POSSIBLE")
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1])
               for i in range(n)]
        ac.lst(ans)
        return

    @staticmethod
    def luogu_p5782(ac=FastIO()):
        n, m = ac.read_list_ints()
        # build_graph|并把索引编码
        edge = [[] for _ in range(4 * n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            edge[a * 2 + 1].append(b * 2)
            edge[b * 2 + 1].append(a * 2)

        # 同一党派内只允许一个人参|
        for i in range(n):
            a, b = 2 * i, 2 * i + 1
            edge[a * 2 + 1].append(b * 2)
            edge[a * 2].append(b * 2 + 1)
            edge[b * 2 + 1].append(a * 2)
            edge[b * 2].append(a * 2 + 1)

        #####################################################
        # 按照强连通缩点
        tarjan = Tarjan(edge)
        for sc in tarjan.scc:
            pre = set()
            for node in sc:
                # 条件相反的两个点不能在一个scc
                if node // 2 in pre:
                    ac.st("NIE")
                    return
                pre.add(node // 2)

        # specific_plan赋予，先出现的确定值
        ans = [0] * 2 * n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        res = [i + 1 for i in range(2 * n) if ans[i]]
        for a in res:
            ac.st(a)
        #####################################################
        # Kosaraju算法
        kosaraju = Kosaraju(4 * n, edge)
        for i in range(2 * n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                ac.st("NIE")
                return
        # 注意是小于符号
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(2 * n)]
        res = [i + 1 for i in range(2 * n) if ans[i]]
        for a in res:
            ac.st(a)
        return

    @staticmethod
    def luogu_4171(ac=FastIO()):
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            # build_graph|并把索引编码
            edge = [[] for _ in range(2 * n)]
            for _ in range(m):
                lst = ac.read_list_strs()
                i = int(lst[0][1:]) - 1
                j = int(lst[1][1:]) - 1
                a = 1 if lst[0][0] == "h" else 0
                b = 1 if lst[1][0] == "h" else 0
                edge[i * 2 + (1 - a)].append(j * 2 + b)
                edge[j * 2 + (1 - b)].append(i * 2 + a)

            #####################################################
            # # 按照强连通缩点
            tarjan = Tarjan(edge)
            ans = True
            for sc in tarjan.scc:
                pre = set()
                for node in sc:
                    # 条件相反的两个点不能在一个scc
                    if node // 2 in pre:
                        ans = False
                    pre.add(node // 2)
            ac.st("GOOD" if ans else "BAD")

            #####################################################
            # Kosaraju算法
            kosaraju = Kosaraju(2 * n, edge)
            ans = True
            for i in range(n):
                if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                    ans = False
                    break
            ac.st("GOOD" if ans else "BAD")
        return
    