import unittest
from collections import defaultdict
from typing import DefaultDict, Set, List, Tuple
from algorithm.src.graph.union_find import UnionFind
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


P1656 炸铁路（https://www.luogu.com.cn/problem/P1656）求割边
P1793 跑步（https://www.luogu.com.cn/problem/P1793）求连通图两个指定点之间的割点，使用枚举与并查集的方式进行求解

F. Is It Flower?（https://codeforces.com/contest/1811/problem/F）无向图求连通分量
"""


class TarjanCC:
    def __init__(self):
        return

    @staticmethod
    def get_strongly_connected_component(n: int, edge: List[Set[int]]) -> Tuple[int, DefaultDict[int, Set[int]], List[int]]:
        """Tarjan求解有向图的强连通分量

        Args:
            n (int): 结点0-n-1
            edge (DefaultDict[int, Set[int]]): 图

        Returns:
            Tuple[int, DefaultDict[int, Set[int]], List[int]]: SCC的数量、分组、每个结点对应的SCC编号
        """

        @FastIO.bootstrap
        def dfs(cur: int) -> None:
            nonlocal dfs_id, scc_id
            if visited[cur]:
                yield
            visited[cur] = True

            order[cur] = low[cur] = dfs_id
            dfs_id += 1
            stack.append(cur)
            in_stack[cur] = True
            for nex in edge[cur]:
                if not visited[nex]:
                    yield dfs(nex)
                    low[cur] = FastIO.min(low[cur], low[nex])
                elif in_stack[nex]:
                    low[cur] = FastIO.min(low[cur], order[nex])  # 注意这里是order

            if order[cur] == low[cur]:
                while stack:
                    top = stack.pop()
                    in_stack[top] = False
                    scc_node_id[scc_id].add(top)
                    node_scc_id[top] = scc_id
                    if top == cur:
                        break
                scc_id += 1
            yield

        dfs_id = 0
        order, low = [FastIO().inf] * n, [FastIO().inf] * n
        visited = [False] * n
        stack = []
        in_stack = [False] * n
        scc_id = 0
        scc_node_id = defaultdict(set)
        node_scc_id = [-1] * n
        for node in range(n):
            if not visited[node]:
                dfs(node)
        return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_cutting_point_and_cutting_edge(n: int, edge: List[Set[int]]) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """Tarjan求解无向图的割点和割边(桥)

        Args:
            n (int): 结点0-n-1
            edge (DefaultDict[int, Set[int]]): 图

        Returns:
            Tuple[Set[int], Set[Tuple[int, int]]]: 割点、桥

        - 边对 (u,v) 中 u < v
        """

        @FastIO.bootstrap
        def dfs(cur: int, parent: int) -> None:
            nonlocal dfs_id
            if visited[cur]:
                return
            visited[cur] = True
            order[cur] = low[cur] = dfs_id
            dfs_id += 1
            dfs_child = 0
            for nex in edge[cur]:
                if nex == parent:
                    continue
                if not visited[nex]:
                    dfs_child += 1
                    yield dfs(nex, cur)
                    low[cur] = FastIO.min(low[cur], low[nex])
                    if low[nex] > order[cur]:
                        cutting_edge.add(tuple(sorted([cur, nex])))
                    if parent != -1 and low[nex] >= order[cur]:
                        cutting_point.add(cur)
                    elif parent == -1 and dfs_child > 1:  # 出发点没有祖先啊，所以特判一下
                        cutting_point.add(cur)
                else:
                    low[cur] = FastIO.min(low[cur], order[nex])  # 注意这里是order
            yield

        dfs_id = 0
        order, low = [FastIO().inf] * n, [FastIO().inf] * n
        visited = [False] * n

        cutting_point = set()
        cutting_edge = set()

        for i in range(n):
            if not visited[i]:
                dfs(i, -1)

        return cutting_point, cutting_edge

    @staticmethod
    def get_point_doubly_connected_component(n: int, edge: List[Set[int]]) -> Tuple[int, DefaultDict[int, Set[int]], List[Set[int]]]:
        """Tarjan求解无向图的点双联通分量

        Args:
            n (int): 结点0-n-1
            edge (DefaultDict[int, Set[int]]): 图

        Returns:
            Tuple[int, DefaultDict[int, Set[int]], List[Set[int]]]: VBCC的数量、分组、每个结点对应的VBCC编号

        - 我们将深搜时遇到的所有边加入到栈里面，
        当找到一个割点的时候，
        就将这个割点往下走到的所有边弹出，
        而这些边所连接的点就是一个点双了

        - 两个点和一条边构成的图也称为(V)BCC,因为两个点均不为割点

        - VBCC编号多余1个的都是割点
        """

        @FastIO.bootstrap
        def dfs(cur: int, parent: int) -> None:
            nonlocal dfs_id, pdcc_id
            if visited[cur]:
                yield
            visited[cur] = True

            order[cur] = low[cur] = dfs_id
            dfs_id += 1

            dfs_child = 0
            for nex in edge[cur]:
                if nex == parent:
                    continue

                if not visited[nex]:
                    dfs_child += 1
                    stack.append((cur, nex))
                    yield dfs(nex, cur)
                    low[cur] = FastIO.min(low[cur], low[nex])

                    # 遇到了割点(根和非根两种)
                    if (parent == -1 and dfs_child > 1) or (parent != -1 and low[nex] >= order[cur]):
                        while stack:
                            top = stack.pop()
                            pdcc_node_id[pdcc_id].add(top[0])
                            pdcc_node_id[pdcc_id].add(top[1])
                            node_pdcc_id[top[0]].add(pdcc_id)
                            node_pdcc_id[top[1]].add(pdcc_id)
                            if top == (cur, nex):
                                break
                        pdcc_id += 1

                elif low[cur] > order[nex]:
                    low[cur] = FastIO.min(low[cur], order[nex])
                    stack.append((cur, nex))
            yield

        dfs_id = 0
        order, low = [FastIO().inf] * n, [FastIO().inf] * n
        visited = [False] * n
        stack = []

        pdcc_id = 0  # 点双个数
        pdcc_node_id = defaultdict(set)  # 每个点双包含哪些点
        node_pdcc_id = [set() for _ in range(n)]  # 每个点属于哪一(几)个点双，属于多个点双的点就是割点

        for node in range(n):
            if not visited[node]:
                dfs(node, -1)
            if stack:
                while stack:
                    top = stack.pop()
                    pdcc_node_id[pdcc_id].add(top[0])
                    pdcc_node_id[pdcc_id].add(top[1])
                    node_pdcc_id[top[0]].add(pdcc_id)
                    node_pdcc_id[top[1]].add(pdcc_id)
                pdcc_id += 1
        return pdcc_id, pdcc_node_id, node_pdcc_id

    def get_edge_doubly_connected_component(self, n: int, edge: List[Set[int]]) -> dict:
        """Tarjan求解无向图的边双联通分量

        Args:
            n (int): 结点0-n-1
            edge (DefaultDict[int, Set[int]]): 图

        Returns:
            Tuple[int, DefaultDict[int, Set[Tuple[int, int]]], DefaultDict[Tuple[int, int], int]]]: EBCC的数量、分组、每条边对应的EBCC编号

        - 边对 (u,v) 中 u < v

        - 实现思路：
          - 将所有的割边删掉剩下的都是边连通分量了(其实可以用并查集做)
          - 处理出割边,再对整个无向图进行一次DFS,对于节点cur的出边(cur,nex),如果它是割边,则跳过这条边不沿着它往下走
        """

        _, cutting_edges = self.get_cutting_point_and_cutting_edge(n, edge)
        for x, y in cutting_edges:
            edge[x].discard(y)
            edge[y].discard(x)
        uf = UnionFind(n)
        for i in range(n):
            for j in edge[i]:
                uf.union(i, j)
        return uf.get_root_part()


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
            edge[i].add(edges[i])
        scc_id, scc_node_id, node_scc_id = TarjanCC().get_strongly_connected_component(n, edge)
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
        edge = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
            edge[x].add(y)
            edge[y].add(x)
        node, _ = TarjanCC().get_cutting_point_and_cutting_edge(n, edge)
        ac.st(len(node))
        ac.lst(sorted([x + 1 for x in node]))
        return

    @staticmethod
    def lg_p8435(ac=FastIO()):
        # 模板: TarjanCC 求无向图点双连通分量
        n, m = ac.read_ints()
        edge = [set() for _ in range(n)]
        degree = [0]*n
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
            # 注意自环的特殊处理
            if x != y:
                edge[x].add(y)
                edge[y].add(x)
                degree[x] += 1
                degree[y] += 1

        pdcc_id, pdcc_node_id, node_pdcc_id = TarjanCC().get_point_doubly_connected_component(n, edge)
        ac.st(len(pdcc_node_id) + sum(degree[i] == 0 for i in range(n)))
        for r in pdcc_node_id:
            ac.lst([len(pdcc_node_id[r])]+[x+1 for x in pdcc_node_id[r]])
        for i in range(n):
            if not degree[i]:
                ac.lst([1, i+1])
        return

    @staticmethod
    def lg_p8436(ac=FastIO()):
        # 模板: TarjanCC 求无向图边双连通分量
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
        group = TarjanCC().get_edge_doubly_connected_component(len(edge), edge)
        res = []
        for r in group:
            lst = [x + 1 for x in group[r] if x < n]
            if lst:
                res.append([len(lst)] + lst)
        ac.st(len(res))
        for a in res:
            ac.lst(a)
        return

    @staticmethod
    def lc_1192_1(n: int, connections: List[List[int]]) -> List[List[int]]:
        # 模板：使用 TarjanCC 求割边
        edge = [set() for _ in range(n)]
        for i, j in connections:
            edge[i].add(j)
            edge[j].add(i)
        cutting_point, cutting_edge = TarjanCC().get_cutting_point_and_cutting_edge(n, edge)
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
