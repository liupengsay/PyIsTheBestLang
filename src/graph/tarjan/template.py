from collections import defaultdict
from math import inf
from typing import DefaultDict, Set, List, Tuple


class TarjanCC:
    def __init__(self):
        return

    @staticmethod
    def get_strongly_connected_component_bfs(n: int, edge: List[List[int]]) \
            -> (int, DefaultDict[int, Set[int]], List[int]):
        # 模板：Tarjan求解有向图的强连通分量 edge为有向边要求无自环与重边
        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        out = []
        in_stack = [0] * n
        scc_id = 0
        scc_node_id = defaultdict(set)
        node_scc_id = [-1] * n
        parent = [-1] * n
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

        # 建立新图
        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in edge[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[a].add(b)
        new_degree = [0] * scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        # SCC的数量，分组，每个结点对应的SCC编号
        return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_point_doubly_connected_component_bfs(n: int, edge: List[List[int]]) \
            -> Tuple[int, DefaultDict[int, Set[int]], List[Set[int]]]:
        # 模板：Tarjan求解无向图的点双连通分量

        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [False] * n
        out = []
        parent = [-1] * n
        group_id = 0  # 点双个数
        group_node = defaultdict(set)  # 每个点双包含哪些点
        node_group_id = [set() for _ in range(n)]  # 每个点属于哪些点双，属于多个点双的点就是割点
        child = [0] * n
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

    def get_edge_doubly_connected_component_bfs(self, n: int, edge: List[Set[int]]) -> List[List[int]]:
        # 模板：Tarjan求解无向图的边双连通分量
        _, cutting_edges = self.get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)

        # 将所有的割边删掉剩下的都是边双连通分量，处理出割边，再对整个无向图进行一次BFS
        visit = [0] * n
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
    def get_cutting_point_and_cutting_edge_bfs(n: int, edge: List[List[int]]) -> (Set[int], List[Tuple[int, int]]):
        # 模板：Tarjan求解无向图的割点和割边（也就是桥）
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        cutting_point = set()
        cutting_edge = []
        child = [0] * n
        parent = [-1] * n
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
