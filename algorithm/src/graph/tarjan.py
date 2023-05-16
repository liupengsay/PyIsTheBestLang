import unittest
from collections import defaultdict, deque
from math import inf
from typing import DefaultDict, Set, List, Tuple
from collections import Counter
from algorithm.src.dp.tree_dp import TreeCentroid
from algorithm.src.fast_io import FastIO

"""
# Tarjan

## 算法功能
Tarjan 算法是基于深度优先搜索的算法，用于求解图的连通性问题，参考[60 分钟搞定图论中的 Tarjan 算法]

- Tarjan 算法可以在线性时间内求出**无向图的割点与桥**，进一步地可以求解**无向图的双连通分量**
- Tarjan 算法可以也可以求解**有向图的强连通分量**，进一步地可以**求有向图的必经点与必经边**

## 可以求有向图与无向图的割点、割边、点双连通分量与边双连通分量
[60 分钟搞定图论中的 Tarjan 算法]: https://zhuanlan.zhihu.com/p/101923309

## 算法伪代码

## 算法模板与测试用例
- 见Tarjan.py

## 经典题目
- 无向有环图求割点[1568. 使陆地分离的最少天数]
- 无向有环图求点最近的环[2204. Distance to a Cycle in Undirected Graph]
- 无向有环图求割边[1192. 查找集群内的「关键连接」]
- 有向有环图求环[2360. 图中的最长环]

===================================力扣===================================
1192. 查找集群内的关键连接（https://leetcode.cn/problems/critical-connections-in-a-network/）求割边
2360. 图中的最长环（https://leetcode.cn/problems/longest-cycle-in-a-graph/solution/by-liupengsay-4ff6/）经典求有向图最长环
[2204. Distance to a Cycle in Undirected Graph]: https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/solution/er-xu-cheng-ming-jiu-xu-zui-python3tarja-09qn/
[1568. 使陆地分离的最少天数]: https://leetcode.cn/problems/minimum-number-of-days-to-disconnect-island/solution/by-liupengsay-zd7w/

===================================洛谷===================================
P3388 【模板】割点（割顶）（https://www.luogu.com.cn/problem/P3388）有自环与重边，求无向图割点
P8435 【模板】点双连通分量（https://www.luogu.com.cn/problem/P8435）有自环与重边，只关注孤立自环即可
P8436 【模板】边双连通分量（https://www.luogu.com.cn/problem/P8436）有自环与重边，通过虚拟节点进行扩边
P2860 [USACO06JAN]Redundant Paths G（https://www.luogu.com.cn/problem/P2860）无向图边双缩点后求树的质心为根时的叶子两两配对数
P2863 [USACO06JAN]The Cow Prom S（https://www.luogu.com.cn/problem/P2863）tarjan求强联通分量

P1656 炸铁路（https://www.luogu.com.cn/problem/P1656）求割边
P1793 跑步（https://www.luogu.com.cn/problem/P1793）求连通图两个指定点之间的割点，使用枚举与并查集的方式进行求解
P2656 采蘑菇（https://www.luogu.com.cn/problem/P2656）使用scc缩点后，计算DAG最长路
P1726 上白泽慧音（https://www.luogu.com.cn/problem/P1726）强连通分量裸题
P2002 消息扩散（https://www.luogu.com.cn/problem/P2002）强连通分量缩点后，计算入度为0的节点个数
P2341 [USACO03FALL/HAOI2006] 受欢迎的牛 G（https://www.luogu.com.cn/problem/P2341）使用scc缩点后计算出度为 0 的点集个数与大小

===================================CodeForces===================================
F. Is It Flower?（https://codeforces.com/contest/1811/problem/F）无向图求连通分量
C. Checkposts（https://codeforces.com/problemset/problem/427/C）有向图的强联通分量进行缩点



"""


class TarjanCC:
    def __init__(self):
        return

    @staticmethod
    def get_strongly_connected_component_bfs(n: int, edge: List[List[int]]) -> (int, DefaultDict[int, Set[int]], List[int]):
        # 模板：Tarjan求解有向图的强连通分量 edge为有向边要求无自环与重边
        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        out = []
        in_stack = [0] * n
        scc_id = 0
        scc_node_id = defaultdict(set)
        node_scc_id = [-1] * n
        parent = [-1]*n
        for node in range(n):
            if not visit[node]:
                stack = [[node, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                        out.append(cur)
                        in_stack[cur] = 1
                    if ind == len(edge[cur]):
                        stack.pop()
                        if order[cur] == low[cur]:
                            while out:
                                top = out.pop()
                                in_stack[top] = 0
                                scc_node_id[scc_id].add(top)
                                node_scc_id[top] = scc_id
                                if top == cur:
                                    break
                            scc_id += 1

                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if not visit[nex]:
                            parent[nex] = cur
                            stack.append([nex, 0])
                        elif in_stack[nex]:
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]  # 注意这里是order
        # SCC的数量，分组，每个结点对应的SCC编号
        return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_point_doubly_connected_component_bfs(n: int, edge: List[List[int]]) -> Tuple[int, DefaultDict[int, Set[int]], List[Set[int]]]:
        # 模板：Tarjan求解无向图的点双连通分量

        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [False] * n
        out = []
        parent = [-1]*n
        group_id = 0  # 点双个数
        group_node = defaultdict(set)  # 每个点双包含哪些点
        node_group_id = [set() for _ in range(n)]  # 每个点属于哪些点双，属于多个点双的点就是割点
        child = [0]*n
        for node in range(n):
            if not visit[node]:
                stack = [[node, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = True
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1

                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            # 遇到了割点（有根和非根两种）
                            if (parent == -1 and child[cur] > 1) or (parent != -1 and low[nex] >= order[cur]):
                                while out:
                                    top = out.pop()
                                    group_node[group_id].add(top[0])
                                    group_node[group_id].add(top[1])
                                    node_group_id[top[0]].add(group_id)
                                    node_group_id[top[1]].add(group_id)
                                    if top == (cur, nex):
                                        break
                                group_id += 1
                            # 我们将深搜时遇到的所有边加入到栈里面，当找到一个割点的时候
                            # 就将这个割点往下走到的所有边弹出，而这些边所连接的点就是一个点双
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            out.append((cur, nex))
                            child[cur] += 1
                            stack.append([nex, 0])
                        elif low[cur] > order[nex]:
                            low[cur] = order[nex]
                            out.append((cur, nex))
            if out:
                while out:
                    top = out.pop()
                    group_node[group_id].add(top[0])
                    group_node[group_id].add(top[1])
                    node_group_id[top[0]].add(group_id)
                    node_group_id[top[1]].add(group_id)
                group_id += 1
        # 点双的数量，点双分组节点，每个结点对应的点双编号（割点属于多个点双）
        return group_id, group_node, node_group_id

    def get_edge_doubly_connected_component_bfs(self, n: int, edge: List[Set[int]]):
        # 模板：Tarjan求解无向图的边双连通分量
        _, cutting_edges = self.get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)

        # 将所有的割边删掉剩下的都是边双连通分量，处理出割边，再对整个无向图进行一次BFS
        visit = [0]*n
        ans = []
        for i in range(n):
            if visit[i]:
                continue
            stack = [i]
            visit[i] = 1
            cur = [i]
            while stack:
                x = stack.pop()
                for j in edge[x]:
                    if not visit[j]:
                        visit[j] = 1
                        stack.append(j)
                        cur.append(j)
            ans.append(cur[:])
        # 边双的节点分组
        return ans

    @staticmethod
    def get_cutting_point_and_cutting_edge_bfs(n: int, edge: List[List[int]]) -> (Set[int], Set[Tuple[int, int]]):
        # 模板：Tarjan求解无向图的割点和割边（也就是桥）
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        cutting_point = set()
        cutting_edge = []
        child = [0]*n
        parent = [-1]*n
        dfs_id = 0
        for i in range(n):
            if not visit[i]:
                stack = [[i, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            pa = parent[cur]
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            if low[nex] > order[cur]:
                                cutting_edge.append((cur, nex) if cur < nex else (nex, cur))
                            if pa != -1 and low[nex] >= order[cur]:
                                cutting_point.add(cur)
                            elif pa == -1 and child[cur] > 1:  # 出发点没有祖先，所以特判一下
                                cutting_point.add(cur)
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            child[cur] += 1
                            stack.append([nex, 0])
                        else:
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]  # 注意这里是order
        # 割点和割边
        return cutting_point, cutting_edge


class TarjanUndirected:
    def __init__(self):
        return

    @staticmethod
    def check_graph(edge, n):
        # edge: 边连接关系 [[],..] n:节点数

        # 访问序号与根节点序号
        visit = [0] * n
        root = [0] * n
        # 割点
        cut_node = []
        # 割边
        cut_edge = []
        # 强连通分量子树
        sub_group = []

        # 中间变量
        stack = []
        index = 1
        in_stack = [0] * n

        def tarjan(i, father):
            nonlocal index
            visit[i] = root[i] = index
            index += 1
            stack.append(i)

            in_stack[i] = 1
            child = 0
            for j in edge[i]:
                if j != father:
                    if not visit[j]:
                        child += 1
                        tarjan(j, i)
                        x, y = root[i], root[j]
                        root[i] = x if x < y else y
                        # 割边 low[i] < dfn[i]
                        if visit[i] < root[j]:
                            cut_edge.append(sorted([i, j]))
                        # 两种情况下才为割点 low[i] <= dfn[i]
                        if father != -1 and visit[i] <= root[j]:
                            cut_node.append(i)
                        elif father == -1 and child >= 2:
                            cut_node.append(i)
                    elif in_stack[j]:
                        x, y = root[i], visit[j]
                        root[i] = x if x < y else y

            if root[i] == visit[i]:
                lst = []
                while stack[-1] != i:
                    lst.append(stack.pop())
                    in_stack[lst[-1]] = 0
                lst.append(stack.pop())
                in_stack[lst[-1]] = 0
                r = min(root[ls] for ls in lst)
                for ls in lst:
                    root[ls] = r
                lst.sort()
                sub_group.append(lst[:])
            return

        for k in range(n):
            if not visit[k]:
                tarjan(k, -1)
        cut_edge.sort()
        cut_node.sort()
        sub_group.sort()
        return cut_edge, cut_node, sub_group


class TarjanDirected:
    def __init__(self):
        return

    @staticmethod
    def check_graph(edge: List[list], n):
        # edge为边连接关系，n为节点数

        # 访问序号与根节点序号
        visit = [0] * n
        root = [0] * n
        # 割点
        cut_node = []
        # 割边
        cut_edge = []
        # 强连通分量子树
        sub_group = []

        # 中间变量
        stack = []
        index = 1

        def tarjan(i):
            nonlocal index
            visit[i] = root[i] = index
            index += 1
            stack.append(i)
            for j in edge[i]:
                if not visit[j]:
                    tarjan(j)
                    root[i] = min(root[i], root[j])
                    if visit[i] < root[j]:
                        cut_edge.append([i, j])
                    if visit[i] <= root[j]:
                        cut_node.append(i)
                elif j in stack:
                    root[i] = min(root[i], visit[j])

            if root[i] == visit[i]:
                lst = []
                while stack[-1] != i:
                    lst.append(stack.pop())
                lst.append(stack.pop())
                r = min(root[ls] for ls in lst)
                for ls in lst:
                    root[ls] = r
                sub_group.append(lst)
            return

        for k in range(n):
            if not visit[k]:
                tarjan(k)

        return cut_edge, cut_node, sub_group


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2360_1(edges: List[int]) -> int:
        # 模板：TarjanCC 求 scc 有向图强连通分量
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
        # 模板：有向图 Tarjan 求 scc 有向图强连通分量
        n = len(edges)
        edge = [[] for _ in range(n)]
        for i in range(n):
            if edges[i] != -1:
                edge[i] = [edges[i]]
        _, _, sub_group = TarjanDirected().check_graph(edge,  n)
        ans = -1
        for sub in sub_group:
            if len(sub) > 1 and len(sub) > ans:
                ans = len(sub)
        return ans

    @staticmethod
    def lg_p3388(ac=FastIO()):
        # 模板: TarjanCC 求无向图割点
        n, m = ac.read_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, _ = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(d) for d in dct])
        cut_node = sorted(list(cut_node))
        ac.st(len(cut_node))
        ac.lst([x+1 for x in cut_node])
        return

    @staticmethod
    def lg_p8435(ac=FastIO()):
        # 模板: TarjanCC 求无向图点双连通分量
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
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
        # 模板: TarjanCC 求无向图边双连通分量
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_ints_minus_one()
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
    def lc_1192_1(n: int, connections: List[List[int]]) -> List[List[int]]:
        # 模板：使用 TarjanCC 求割边
        edge = [set() for _ in range(n)]
        for i, j in connections:
            edge[i].add(j)
            edge[j].add(i)
        cutting_point, cutting_edge = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        return [list(e) for e in cutting_edge]

    @staticmethod
    def lc_1192_2(n: int, connections: List[List[int]]) -> List[List[int]]:
        # 模板：使用 Tarjan 求割边
        edge = [[] for _ in range(n)]
        for i, j in connections:
            edge[i].append(j)
            edge[j].append(i)
        cut_edge, cut_node, sub_group = TarjanUndirected().check_graph(edge, n)
        return cut_edge

    @staticmethod
    def lg_p1656(ac=FastIO()):
        # 模板：tarjan求无向图割边
        n, m = ac.read_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)
        cut_node, cut_edge = TarjanCC().get_cutting_point_and_cutting_edge_bfs(n, [list(d) for d in dct])
        cut_edge = sorted([list(e) for e in cut_edge])
        for x in cut_edge:
            ac.lst([w + 1 for w in x])
        return

    @staticmethod
    def lg_p2860(ac=FastIO()):
        # 模板: TarjanCC 求无向图边双连通分量进行缩点后，计算质心为根时的叶子数
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        degree = defaultdict(int)
        pre = set()
        for _ in range(m):
            a, b = ac.read_ints_minus_one()
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
        center = TreeCentroid().get_tree_centroid(dct)
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
        ac.st((ans+1)//2)
        return

    @staticmethod
    def lg_p2863(ac=FastIO()):
        # 模板: TarjanCC 求强连通分量
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_ints_minus_one()
            edge[a].add(b)
        _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in edge])

        ans = 0
        for g in group:
            ans += len(group[g]) > 1
        ac.st(ans)
        return

    @staticmethod
    def cf_427c(ac=FastIO()):
        # 模板：tarjan进行有向图缩点后计数
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(ac.read_int()):
            x, y = ac.read_ints_minus_one()
            dct[x].append(y)
        _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
        ans = 1
        cost = 0
        mod = 10**9+7
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
        # 模板：使用scc缩点后，计算DAG最长路
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        edges = []
        for _ in range(m):
            a, b, c, d = ac.read_list_strs()
            a = int(a)-1
            b = int(b)-1
            c = int(c)
            d = float(d)
            edges.append([a, b, c, d])
            if a != b:
                edge[a].add(b)
        s = ac.read_int()-1
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(x) for x in edge])
        cnt = defaultdict(int)
        dis = [defaultdict(int) for _ in range(scc_id)]
        pre = [set() for _ in range(scc_id)]
        for i, j, c, d in edges:
            if node_scc_id[i] == node_scc_id[j]:
                x = 0
                while c:
                    x += c
                    c = int(c*d)
                cnt[node_scc_id[i]] += x
            else:
                a, b = node_scc_id[i], node_scc_id[j]
                dis[a][b] = ac.max(dis[a][b], c)
                pre[b].add(a)

        # 注意这里可能有 0 之外的入度为 0 的点，需要先进行拓扑消除
        stack = deque([i for i in range(scc_id) if not pre[i] and i != node_scc_id[s]])
        while stack:
            i = stack.popleft()
            for j in dis[i]:
                pre[j].discard(i)
                if not pre[j]:
                    stack.append(j)

        # 广搜计算最长路，进一步还可以确定相应的具体路径
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
        # 模板：强连通分量裸题
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, t = ac.read_ints_minus_one()
            dct[a].append(b)
            if t == 1:
                dct[b].append(a)
        _, scc_node_id, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
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
        # 模板：强连通分量缩点后，计算入度为0的节点个数
        n, m = ac.read_ints()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            if i != j:
                dct[i].add(j)
        scc_id, _, node_scc_id = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in dct])

        # 计算新图
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
        # 模板：使用scc缩点后计算出度为 0 的点集个数与大小
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_ints()
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


class TestGeneral(unittest.TestCase):
    def test_undirected_graph(self):
        # 无向无环图
        edge = [[1, 2], [0, 3], [0, 3], [1, 2]]
        n = 4
        ta = TarjanUndirected()
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert not cut_edge
        assert not cut_node
        assert sub_group == [[0, 1, 2, 3]]

        # 无向有环图
        edge = [[1, 2, 3], [0, 2], [0, 1], [0]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 3]]
        assert cut_node == [0]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有自环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2, 3]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]
        return

    def test_directed_graph(self):
        # 有向无环图
        edge = [[1, 2], [], [3], []]
        n = 4
        ta = Tarjan()
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 1], [0, 2], [2, 3]]
        assert cut_node == [0, 2]
        assert sub_group == [[0], [1], [2], [3]]

        edge = [[1, 2], [2], [3], []]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 1], [1, 2], [2, 3]]
        assert cut_node == [1, 2]
        assert sub_group == [[0], [1], [2], [3]]

        # 有向有环图
        edge = [[1, 2], [2], [0, 3], []]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]
        return


if __name__ == '__main__':
    unittest.main()
