"""

Algorithm：topological_sorting、内向基环树（有向或者无向，连通块有k个节点以及k条边）、bfs序、拓扑序
Function：有向图sorting，无向图在选定根节点的情况下也可以topological_sorting
内向基环树介绍：https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/solution/nei-xiang-ji-huan-shu-tuo-bu-pai-xu-fen-c1i1b/

====================================LeetCode====================================
360（https://leetcode.com/problems/longest-cycle-in-a-graph/）topological_sorting有向图内向基环树最长环
2392（https://leetcode.com/problems/build-a-matrix-with-conditions/）分别通过行列的topological_sorting来确定数字所在索引，数字可能相同，需要union_find
2371（https://leetcode.com/problems/minimize-maximum-value-in-a-grid/）分别通过行列的topological_sorting来确定数字所在索引，数字都不同可以greedy
2127（https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/）topological_sorting确定DAG内向基环，按照环的大小classification_discussion
127（https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/）
269（https://leetcode.com/problems/alien-dictionary/）按照lexicographical_order建图，与topological_sorting的应用
2603（https://leetcode.com/contest/weekly-contest-338/problems/collect-coins-in-a-tree/）无向图topological_sorting内向基环树
2204（https://leetcode.com/problems/distance-to-a-cycle-in-undirected-graph/https://leetcode.com/problems/distance-to-a-cycle-in-undirected-graph/）无向图topological_sorting
1857（https://leetcode.com/problems/largest-color-value-in-a-directed-graph/）topological_sortingDP
1932（https://leetcode.com/problems/range_merge_to_disjoint-bsts-to-create-single-bst/）连通性、topological_sorting与二叉搜索树判断
1591（https://leetcode.com/contest/biweekly-contest-35/problems/strange-printer-ii/）建图判断topological_sorting是否无环
2192（https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/）有向图DAGtopological_sorting


=====================================LuoGu======================================
1960（https://www.luogu.com.cn/problem/P1960）topological_sorting是否唯一
1992（https://www.luogu.com.cn/problem/P1992）topological_sorting有向图是否有环
2712（https://www.luogu.com.cn/problem/P2712）topological_sorting非环节点数
6145（https://www.luogu.com.cn/problem/P6145）topological_sorting每个节点最晚的访问时间点
1137（https://www.luogu.com.cn/problem/P1137）topological_sorting，可达的最长距离
1347（https://www.luogu.com.cn/problem/P1347）topological_sorting确定lexicographical_order与矛盾或者无唯一解
1685（https://www.luogu.com.cn/problem/P1685）DAGtopological_sortingDP路径条数与耗时
3243（https://www.luogu.com.cn/problem/P3243）反向建图topological_sorting结合二叉heapq顺序implemention
5536（https://www.luogu.com.cn/problem/P5536）无向图topological_sorting从外到内消除最外圈的节点
6037（https://www.luogu.com.cn/problem/P6037）无向图基环树union_findtopological_sorting与环implemention
6255（https://www.luogu.com.cn/problem/P6255）简单无向图union_find连通块后topological_sorting寻找环的信息
6417（https://www.luogu.com.cn/problem/P6417）有向图基环树greedy应用topological_sorting由外向内
6560（https://www.luogu.com.cn/problem/P6560）反向建图topological_sorting与博弈必胜态
8655（https://www.luogu.com.cn/problem/P8655）topological_sorting有向基环树的环
8943（https://www.luogu.com.cn/problem/P8943）无向图基环树博弈

===================================CodeForces===================================
1454E（https://codeforces.com/contest/1454/problem/E）基环树counterbrute_force
1907G（https://codeforces.com/contest/1907/problem/G）directed_circle_based_tree|greedy|implemention|topological_sort

====================================AtCoder=====================================
F - Well-defined Path Queries on a Namori（https://atcoder.jp/contests/abc266/）（无向图的内向基环树，求简单路径的树枝连通）
最喜欢的数列（https://www.hackerrank.com/challenges/favourite-sequence/problem?isFullScreen=true）topological and heap for minimum lexi order

=====================================AcWing=====================================
3696（https://www.acwing.com/problem/content/description/3699/）bfs序即拓扑序与DAGconstruction
3828（https://www.acwing.com/problem/content/description/3831/）有向图DAGtopological_sortingDP模板题并判断有无环
4626（https://www.acwing.com/problem/content/description/4629/）有向图内向基环树判断每个环的大小

"""

import copy
import math
from collections import defaultdict, deque
from heapq import heapify, heappop, heappush
from math import inf
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_266f(ac=FastIO()):
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        uf = UnionFind(n)

        degree = [0] * n
        for _ in range(n):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)
            degree[u] += 1
            degree[v] += 1

        que = deque()
        for i in range(n):
            if degree[i] == 1:
                que.append(i)
        while que:
            now = que.popleft()
            nex = edge[now][0]
            degree[now] -= 1
            degree[nex] -= 1
            edge[nex].remove(now)
            uf.union(now, nex)
            if degree[nex] == 1:
                que.append(nex)

        q = ac.read_int()
        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if uf.is_connected(x, y):
                ac.st("Yes")
            else:
                ac.st("No")
        return

    @staticmethod
    def ac_3696(ac=FastIO()):
        # bfs序即拓扑序与DAGconstruction
        for _ in range(ac.read_int()):
            def check():
                n, m = ac.read_list_ints()
                dct = [[] for _ in range(n)]
                degree = [0] * n
                edges = []
                ans = []
                for _ in range(m):
                    t, x, y = ac.read_list_ints()
                    x -= 1
                    y -= 1
                    if t == 1:
                        ans.append([x, y])
                        dct[x].append(y)
                        degree[y] += 1
                    else:
                        edges.append([x, y])

                # topological_sorting判断有向图是否存在环，同时记录节点的拓扑顺序
                order = [0] * n
                stack = [i for i in range(n) if degree[i] == 0]
                ind = 0
                while stack:
                    nex = []
                    for i in stack:
                        order[i] = ind
                        ind += 1
                        for j in dct[i]:
                            degree[j] -= 1
                            if degree[j] == 0:
                                nex.append(j)
                    stack = nex[:]
                if any(degree[x] > 0 for x in range(n)):
                    ac.st("NO")
                    return

                # 按照拓扑序依次给与方向
                ac.st("YES")
                for x, y in edges:
                    if order[x] < order[y]:
                        ac.lst([x + 1, y + 1])
                    else:
                        ac.lst([y + 1, x + 1])
                for x, y in ans:
                    ac.lst([x + 1, y + 1])
                return

            check()
        return

    @staticmethod
    def lc_2360(edges: List[int]) -> int:
        # topological_sorting有向图内向基环树最长环
        n = len(edges)
        # 记录入度
        degree = defaultdict(int)
        for i in range(n):
            if edges[i] != -1:
                degree[edges[i]] += 1

        # 先消除无环部分
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                if edges[i] != -1:
                    degree[edges[i]] -= 1
                    if not degree[edges[i]]:
                        nex.append(edges[i])
            stack = nex

        # 注意没有自环
        visit = [int(degree[i] == 0) for i in range(n)]
        ans = -1
        for i in range(n):
            cnt = 0
            while not visit[i]:
                visit[i] = 1
                cnt += 1
                i = edges[i]
            if cnt and cnt > ans:
                ans = cnt
        return ans

    @staticmethod
    def lc_2392(k: int, row_conditions: List[List[int]], col_conditions: List[List[int]]) -> List[List[int]]:

        # 行列topological_sortingconstruction矩阵
        def check(cond):
            dct = defaultdict(list)
            degree = defaultdict(int)
            for i, j in cond:
                dct[i].append(j)
                degree[j] += 1
            stack = [i for i in range(1, k + 1) if not degree[i]]
            ans = []
            while stack:
                ans.extend(stack)
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if not degree[j]:
                            nex.append(j)
                stack = nex
            return ans

        row = check(row_conditions)
        col = check(col_conditions)
        if len(row) != k or len(col) != k:
            return []

        row_ind = {row[i]: i for i in range(k)}
        col_ind = {col[i]: i for i in range(k)}
        res = [[0] * k for _ in range(k)]
        for i in range(1, k + 1):
            res[row_ind[i]][col_ind[i]] = i
        return res

    @staticmethod
    def lg_p1137(ac=FastIO()):
        # topological_sorting最长链条
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints()
            degree[j - 1] += 1
            dct[i - 1].append(j - 1)
        cnt = [1] * n
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            i = stack.pop()
            for j in dct[i]:
                cnt[j] = max(cnt[j], cnt[i] + 1)
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
        for a in cnt:
            ac.st(a)
        return

    @staticmethod
    def lg_p1347(ac=FastIO()):
        # topological_sorting确定lexicographical_order与矛盾或者无唯一解
        n, m = ac.read_list_ints()
        dct_ = defaultdict(list)
        degree_ = defaultdict(int)

        def check(dct, degree, nodes):
            # 每次都判断结果
            stack = [k for k in nodes if not degree[k]]

            res = []
            unique = True
            while stack:
                res.extend(stack)
                if len(stack) > 1:
                    unique = False
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if not degree[j]:
                            nex.append(j)
                stack = nex
            # 稳定的topological_sorting
            if unique and len(res) == n:
                ss = "".join(res)
                return True, f"Sorted sequence determined after {x} relations: {ss}."
            # 存在环
            if len(res) < m:
                return True, f"Inconsistency found after {x} relations."
            # 不稳定的topological_sorting
            return False, "Sorted sequence cannot be determined."

        nodes_ = set()
        res_ans = ""
        for x in range(1, m + 1):
            s = input().strip()
            if res_ans:
                continue
            dct_[s[0]].append(s[2])
            degree_[s[2]] += 1
            nodes_.add(s[0])
            nodes_.add(s[2])
            m = len(nodes_)
            flag, ans = check(copy.deepcopy(dct_), copy.deepcopy(degree_), copy.deepcopy(nodes_))
            if flag:
                res_ans = ans
        if res_ans:
            print(res_ans)
        else:
            print("Sorted sequence cannot be determined.")
        return

    @staticmethod
    def lg_p1685(ac=FastIO()):
        # topological_sorting经过每条边的路径条数
        n, m, s, e, t = ac.read_list_ints()
        s -= 1
        e -= 1
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            degree[j] += 1
        mod = 10000

        # 记录总时间与路径条数
        time = [0] * n
        cnt = [0] * n
        stack = [s]
        cnt[s] = 1
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
                # 更新 i 到 j 的路径条数与相应耗时
                cnt[j] += cnt[i]
                time[j] += cnt[i] * w + time[i]
                time[j] %= mod
        # 减去回头的时间
        ans = time[e] + (cnt[e] - 1) * t
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3243(ac=FastIO()):
        # 反向建图topological_sorting结合二叉heapq顺序implemention
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            degree = [0] * n
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                dct[j].append(i)
                degree[i] += 1
            ans = []
            stack = [-i for i in range(n) if not degree[i]]
            heapify(stack)
            while stack:
                # 优先选择入度为 0 且编号最大的
                i = -heappop(stack)
                ans.append(i)
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        heappush(stack, -j)
            if len(ans) == n:
                # 翻转后则lexicographical_order最小
                ans.reverse()
                ac.lst([x + 1 for x in ans])
            else:
                ac.st("Impossible!")
        return

    @staticmethod
    def lg_p5536_1(ac=FastIO()):
        # 直径的greedy方式选取以树的直径中点向外辐射的节点
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        x, y, _, path = TreeDiameterInfo().get_diameter_info(dct)

        # 树形 DP 每个节点的深度与子树最大节点深度
        root = path[len(path) // 2]
        deep = [0] * n
        max_deep = [0] * n
        stack = [[root, -1, 0]]
        while stack:
            i, fa, d = stack.pop()
            if i >= 0:
                stack.append([~i, fa, d])
                deep[i] = d
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, d + 1])
            else:
                i = ~i
                max_deep[i] = deep[i]
                for j in dct[i]:
                    if j != fa:
                        max_deep[i] = ac.max(max_deep[i], max_deep[j])

        # 选取 k 个节点后剩下的节点的最大值
        lst = [max_deep[i] - deep[i] for i in range(n)]
        lst.sort(reverse=True)
        ac.st(lst[k] + 1)
        return

    @staticmethod
    def lg_p5536_2(ac=FastIO()):

        # 无向图topological_sorting从外到内消除最外圈的节点
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1

        # 按照度为 1  BFS 消除
        rem = n - k
        ans = 0
        stack = deque([[i, 1] for i in range(n) if degree[i] == 1])
        while rem:
            i, d = stack.popleft()
            ans = d
            rem -= 1
            for j in dct[i]:
                if degree[j] > 1:
                    degree[j] -= 1
                    if degree[j] == 1:
                        stack.append([j, d + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6037(ac=FastIO()):
        # 无向图基环树union_findtopological_sorting与环implemention
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        # 首先分割连通分量
        uf = UnionFind(n)
        degree = [0] * n
        edge = []
        for _ in range(n):
            u, v, w, p = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w, p])
            dct[v].append([u, w, p])
            uf.union(u, v)
            edge.append([u, v, w])
            degree[u] += 1
            degree[v] += 1
        # 其次对每个分量结果
        part = uf.get_root_part()
        ans = [-1] * n
        for p in part:
            # topological_sorting找出环
            stack = deque([i for i in part[p] if degree[i] == 1])
            visit = set()
            while stack:
                i = stack.popleft()
                visit.add(i)
                for j, _, _ in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 1:
                        stack.append(j)
            # 根据环greedy结果
            circle = [i for i in part[p] if i not in visit]
            s = sum(w for i, _, w in edge if uf.find(i) == p)
            for x in circle:
                cur = [[j, w, p] for j, w, p in dct[x] if degree[j] > 1]
                cur.sort(key=lambda it: it[-1])
                ans[x] = s - cur[0][1]
            stack = deque(circle)
            while stack:
                i = stack.popleft()
                for j, _, _ in dct[i]:
                    if ans[j] == -1:
                        ans[j] = ans[i]
                        stack.append(j)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p6255(ac=FastIO()):
        # 简单无向图union_find连通块后topological_sorting寻找环的信息
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        uf = UnionFind(n)
        for _ in range(m):  # 简单无向图即没有环套环
            i, j = ac.read_list_ints_minus_one()
            degree[j] += 1
            degree[i] += 1
            dct[i].append(j)
            dct[j].append(i)
            uf.union(i, j)
        # 连通块
        part = uf.get_root_part()
        ans = []
        for p in part:
            lst = part[p]
            # topological_sorting找环
            nodes = [i for i in lst if degree[i] == 1]
            stack = nodes[:]
            visit = set()
            cnt = 0
            while stack:
                cnt += len(stack)
                for i in stack:
                    visit.add(i)
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if degree[j] == 1:
                            nex.append(j)
                stack = nex[:]
            if cnt == len(part[p]):
                # 没有环则所有外围点出发的边都是死路
                for i in nodes:
                    for j in dct[i]:
                        ans.append([i + 1, j + 1])
            else:
                # 有环则所有环上的点
                for i in lst:
                    if i not in visit:
                        for j in dct[i]:
                            if j in visit:
                                ans.append([i + 1, j + 1])
        ans.sort()
        ac.st(len(ans))
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p6417(ac=FastIO()):
        # 有向图基环树greedy应用topological_sorting由外向内
        n = ac.read_int()
        dct = [ac.read_int() - 1 for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            degree[dct[i]] += 1
        # 外层直接作为坏蛋
        stack = [x for x in range(n) if not degree[x]]
        visit = [-1] * n
        ans = len(stack)
        for i in stack:
            visit[i] = 1
        while stack:
            nex = []
            for i in stack:
                degree[dct[i]] -= 1
                # 确定坏蛋或者是平民的角色后根据度角色指定
                if (not degree[dct[i]] or visit[i] == 1) and visit[dct[i]] == -1:
                    if visit[i] == 1:
                        # 父亲是坏蛋必然是平民
                        visit[dct[i]] = 0
                    else:
                        # 入度为 0 优先指定为坏蛋
                        ans += 1
                        visit[dct[i]] = 1
                    nex.append(dct[i])
            stack = nex[:]

        for i in range(n):
            x = 0
            # 剩余环的大小
            while visit[i] == -1:
                visit[i] = 1
                x += 1
                i = dct[i]
            # 环内的坏蛋最多个数
            ans += x // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p6560(ac=FastIO()):
        # 反向建图topological_sorting与博弈必胜态
        n, m, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [[0, -1, 0] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[j].append(i)
            degree[i][0] += 1

        visit = [[-1, 0] for _ in range(n)]
        for ind in range(q):
            s, e = ac.read_list_ints_minus_one()
            visit[s] = [ind, 0]
            stack = deque([x for x in range(n) if not degree[x][0] or x == e])
            for i in stack:
                visit[i] = [ind, -1]
            while stack and not visit[s][1]:
                i = stack.popleft()
                for j in dct[i]:
                    if visit[j][0] != ind:
                        visit[j] = [ind, 0]
                    if degree[j][1] != ind:
                        degree[j][1] = ind
                        degree[j][2] = degree[j][0]
                    if visit[j][1]:
                        continue
                    degree[j][2] -= 1
                    if visit[i][1] == -1:
                        visit[j][1] = 1
                        stack.append(j)
                    elif not degree[j][2] and visit[i][1] == 1:
                        visit[j][1] = -1
                        stack.append(j)
            ac.st(visit[s][1])
        return

    @staticmethod
    def lg_p8655(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = [x for x in range(n) if degree[x] == 1]
        while stack:
            y = stack.pop()
            for z in dct[y]:
                degree[z] -= 1
                if degree[z] == 1:
                    stack.append(z)
        ans = [x + 1 for x in range(n) if degree[x] == 2]
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8943(ac=FastIO()):
        # 无向图基环树博弈
        n, q = ac.read_list_ints()
        degree = [0] * n
        dct = [[] for _ in range(n)]
        for _ in range(n):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
            degree[x] += 1
            degree[y] += 1

        # 找出环
        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        dis = [inf] * n
        stack = [i for i in range(n) if degree[i] > 1]
        path = [[stack[0], -1]]
        while True:
            for j in dct[path[-1][0]]:
                if degree[j] > 1 and j != path[-1][1]:
                    path.append([j, path[-1][0]])
                    break
            if path[-1][0] == path[0][0]:
                path.pop()
                break
        path = [p for p, _ in path]
        ind = {num: i for i, num in enumerate(path)}

        # 每个点到环上的祖先节点与距离
        parent = [-1] * n
        for i in stack:
            parent[i] = i
        d = 0
        while stack:
            nex = []
            for i in stack:
                dis[i] = d
            for i in stack:
                for j in dct[i]:
                    if dis[j] == inf:
                        parent[j] = parent[i]
                        nex.append(j)
            stack = nex[:]
            d += 1

        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if dis[x] == 0:
                ac.st("Survive" if x != y else "Deception")
                continue

            a = parent[x]
            b = parent[y]
            # 只有先到达环才可能幸免
            dis_y = dis[y] + ac.min(len(path) - abs(ind[a] - ind[b]), abs(ind[a] - ind[b]))
            if dis[x] < dis_y:
                ac.st("Survive")
            else:
                ac.st("Deception")
        return

    @staticmethod
    def lc_2127(favorite: List[int]) -> int:
        # topological_sorting确定DAG内向基环，按照环的大小classification_discussion
        n = len(favorite)
        degree = [0] * n
        for i in range(n):
            degree[favorite[i]] += 1
        depth = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        while stack:
            nex = []
            for i in stack:
                j = favorite[i]
                degree[j] -= 1
                a, b = depth[i] + 1, depth[j]
                depth[j] = a if a > b else b
                if not degree[j]:
                    nex.append(j)
            stack = nex[:]
        ans = 0
        bicycle = 0
        for i in range(n):
            if not degree[i]:
                continue
            lst = [i]
            degree[i] = 0
            x = favorite[i]
            while x != i:
                lst.append(x)
                degree[x] = 0
                x = favorite[x]
            if len(lst) == 2:
                # 一种是所有的2元环外接链拼接起来
                bicycle += depth[lst[0]] + depth[lst[1]] + 2
            elif len(lst) > ans:
                # 一种是只有一个大于2的环
                ans = len(lst)

        ans = ans if ans > bicycle else bicycle
        return ans

    @staticmethod
    def lc_2192(n: int, edges: List[List[int]]) -> List[List[int]]:
        # 有向图DAGtopological_sorting
        ans = [set() for _ in range(n)]
        degree = [0] * n
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            degree[j] += 1
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    for x in ans[i]:
                        ans[j].add(x)
                    ans[j].add(i)
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return [sorted(list(a)) for a in ans]

    @staticmethod
    def lc_2204(n: int, edges: List[List[int]]) -> List[int]:
        # 无向图topological_sorting
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        circle = deque([i for i in range(n) if degree[i] > 1])
        ans = [-1] * n
        for i in circle:
            ans[i] = 0
        while circle:
            i = circle.popleft()
            for j in dct[i]:
                if ans[j] == -1:
                    ans[j] = ans[i] + 1
                    circle.append(j)
        return ans

    @staticmethod
    def lc_1857(colors: str, edges: List[List[int]]) -> int:

        # topological_sortingDP
        n = len(colors)
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            if i == j:
                return -1
            dct[i].append(j)
            degree[j] += 1

        cnt = [[0] * 26 for _ in range(n)]
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            cnt[i][ord(colors[i]) - ord("a")] += 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                for c in range(26):
                    a, b = cnt[j][c], cnt[i][c]
                    cnt[j][c] = a if a > b else b
                if not degree[j]:
                    cnt[j][ord(colors[j]) - ord("a")] += 1
                    stack.append(j)
        if not all(x == 0 for x in degree):
            return -1
        return max(max(c) for c in cnt)

    @staticmethod
    def lc_1932(trees: List[TreeNode]) -> Optional[TreeNode]:
        # 连通性、topological_sorting与二叉搜索树判断

        nodes = set()
        dct = defaultdict(list)
        degree = defaultdict(int)
        for root in trees:

            def dfs(node):
                if not node:
                    return
                nodes.add(node.val)
                if node.left:
                    xxx = node.left.val
                    dct[node.val].append(xxx)
                    degree[xxx] += 1
                    dfs(node.left)
                if node.right:
                    xxx = node.right.val
                    dct[node.val].append(xxx)
                    degree[xxx] += 1
                    dfs(node.right)
                return

            dfs(root)

        nodes = list(nodes)
        m = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}

        # 连通性
        uf = UnionFind(m)
        for x in dct:
            for y in dct[x]:
                uf.union(ind[x], ind[y])
        if uf.part != 1:
            return

        # 二叉性与topological_sorting唯一根
        for num in nodes:
            if len(dct[num]) > 2:
                return
        stack = [num for num in nodes if not degree[num]]
        if len(stack) != 1:
            return
        r = stack[0]
        while stack:
            nex = []
            for x in stack:
                for y in dct[x]:
                    degree[y] -= 1
                    if not degree[y]:
                        nex.append(y)
            stack = nex[:]
        if not all(degree[x] == 0 for x in nodes):
            return

        # 二叉搜索特性

        def dfs(w, floor, ceil):
            nonlocal ans
            if not ans:
                return
            if not floor < w < ceil:
                ans = False
                return
            node = TreeNode(w)
            for z in dct[w]:
                if z < w:
                    node.left = dfs(z, floor, w)
                else:
                    node.right = dfs(z, w, ceil)
            return node

        ans = True
        root = dfs(r, -inf, inf)

        if ans:
            return root
        return

    @staticmethod
    def ac_3828(ac=FastIO()):
        # 有向图DAGtopological_sortingDP模板题并判断有无环
        m, n = ac.read_list_ints()
        ind = {w: i for i, w in enumerate("QWER")}
        grid = [ac.read_str() for _ in range(m)]

        # 建图
        dct = [[] for _ in range(m * n)]
        degree = [0] * (m * n)
        for i in range(m):
            for j in range(n):
                x = ind[grid[i][j]]
                for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= a < m and 0 <= b < n:
                        y = ind[grid[a][b]]
                        if y == (x + 1) % 4:
                            dct[i * n + j].append(a * n + b)
                            degree[a * n + b] += 1

        # topological_sortingDP
        pre = [0] * (m * n)
        stack = [i for i in range(m * n) if not degree[i]]
        for i in stack:
            if grid[i // n][i % n] == "Q":
                pre[i] = 1
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if pre[i] > pre[j]:
                        pre[j] = pre[i]
                    if not degree[j]:
                        if not pre[j]:
                            if grid[j // n][j % n] == "Q":
                                pre[j] = 1
                        else:
                            pre[j] += 1
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            ac.st("infinity")
            return
        ans = max(x // 4 for x in pre)
        ac.st(ans if ans else "none")
        return

    @staticmethod
    def ac_4626(ac=FastIO()):
        # 有向图内向基环树判断每个环的大小
        n = ac.read_int()
        a = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            dct[i].append(a[i])
            degree[a[i]] += 1
        # 全是环上的点才行
        if any(d == 0 for d in degree):
            ac.st(-1)
            return
        # BFS判断每个环的大小并累计结果
        ans = 1
        for i in range(n):
            if degree[i] == 0:
                continue
            stack = [i]
            degree[i] = 0
            cur = 1
            while stack:
                x = stack.pop()
                for j in dct[x]:
                    if degree[j] > 0:
                        degree[j] = 0
                        cur += 1
                        stack.append(j)
            if cur % 2:
                ans = math.lcm(cur, ans)
            else:
                ans = math.lcm(cur // 2, ans)
        ac.st(ans)
        return

    @staticmethod
    def lc_1591(grid: List[List[int]]) -> bool:
        # 建图判断topological_sorting是否无环
        color = defaultdict(list)
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                color[grid[i][j]].append([i, j])

        pos = defaultdict(list)
        for c in color:
            lst = color[c]
            x1 = min(x for x, _ in lst)
            x2 = max(x for x, _ in lst)

            y1 = min(x for _, x in lst)
            y2 = max(x for _, x in lst)
            pos[c] = [x1, x2, y1, y2]

        degree = defaultdict(int)
        dct = defaultdict(list)
        for x in pos:
            a1, a2, b1, b2 = pos[x]
            for y in pos:
                if x != y:
                    for a, b in color[y]:
                        if a1 <= a <= a2 and b1 <= b <= b2:
                            dct[x].append(y)
                            degree[y] += 1
        stack = [x for x in pos if not degree[x]]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex[:]
        return all(degree[x] == 0 for x in pos)