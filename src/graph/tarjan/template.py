from collections import defaultdict

from src.utils.fast_io import inf


class TarjanCC:
    def __init__(self):
        return

    @staticmethod
    def get_strongly_connected_component_bfs(n: int, edge):
        assert all(i not in edge[i] for i in range(n))
        assert all(len(set(edge[i])) == len(edge[i]) for i in range(n))
        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        out = []
        in_stack = [0] * n
        scc_id = 0
        # nodes list of every scc_id part
        scc_node_id = defaultdict(set)
        # index if original node and value is scc_id part
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
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]

        # new graph after scc
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
        return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_point_doubly_connected_component_bfs(n: int, edge):

        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [False] * n
        out = []
        parent = [-1] * n
        # number of group
        group_id = 0
        # nodes list of every group part
        group_node = defaultdict(set)
        # index is original node and value is group_id set
        # cut node belong to two or more group
        node_group_id = [set() for _ in range(n)]
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
                            # cut node with rooted or not-rooted
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
                            # We add all the edges encountered during deep search to the stack
                            # and when we find a cut point
                            # Pop up all the edges that this cutting point goes down to
                            # and the points connected by these edges are a pair of dots
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
        return group_id, group_node, node_group_id

    def get_edge_doubly_connected_component_bfs(self, n: int, edge):
        _, cutting_edges = self.get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)
        # Remove all cut edges and leaving only edge doubly connected components
        # process the cut edges and then perform bfs on the entire undirected graph
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
        # group of nodes
        return ans

    @staticmethod
    def get_cutting_point_and_cutting_edge_bfs(n: int, edge):
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
                            elif pa == -1 and child[cur] > 1:
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
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]
        return cutting_point, cutting_edge


class TarjanUndirected:
    def __init__(self):
        return

    @staticmethod
    def check_graph(edge, n):
        visit = [0] * n
        root = [0] * n
        cut_node = []
        cut_edge = []
        sub_group = []
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
                        # cut edge where low[i] < dfn[i]
                        if visit[i] < root[j]:
                            cut_edge.append(sorted([i, j]))
                        # cur point where low[i] <= dfn[i]
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
    def check_graph(edge, n):
        visit = [0] * n
        root = [0] * n
        cut_node = []
        cut_edge = []
        sub_group = []
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
