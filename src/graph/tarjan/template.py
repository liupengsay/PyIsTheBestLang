import math


class DirectedGraphForTarjanScc:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * self.n
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.node_scc_id = [0]
        self.edge_id = 1
        self.scc_id = 0
        self.original_edge = set()
        return

    def initialize_graph(self):
        self.point_head = [0] * self.n
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.original_edge = set()
        return

    def add_directed_original_edge(self, i, j):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.original_edge.add(i * self.n + j)
        return

    def add_directed_edge(self, i, j):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.edge_id += 1
        return

    def build_scc(self):
        for val in self.original_edge:
            i, j = val // self.n, val % self.n
            if i != j:
                self.add_directed_edge(i, j)
        dfs_id = 0
        order = [self.n] * self.n
        low = [self.n] * self.n
        visit = [0] * self.n
        out = []
        in_stack = [0] * self.n
        self.node_scc_id = [-1] * self.n
        parent = [-1] * self.n
        point_head = self.point_head[:]
        for node in range(self.n):
            if not visit[node]:
                stack = [node]
                while stack:
                    cur = stack[-1]
                    ind = point_head[cur]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                        out.append(cur)
                        in_stack[cur] = 1
                    if not ind:
                        stack.pop()
                        if order[cur] == low[cur]:
                            while out:
                                top = out.pop()
                                in_stack[top] = 0
                                self.node_scc_id[top] = self.scc_id
                                if top == cur:
                                    break
                            self.scc_id += 1

                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = min(low[cur], low[nex])
                    else:
                        nex = self.edge_to[ind]
                        point_head[cur] = self.edge_next[ind]
                        if not visit[nex]:
                            parent[nex] = cur
                            stack.append(nex)
                        elif in_stack[nex]:
                            low[cur] = min(low[cur], order[nex])
        # topological_order is [self.scc_id-1,self.scc_id-2,...,0]
        return

    def get_scc_edge_degree(self):
        scc_edge = set()
        for i in range(self.n):
            ind = self.point_head[i]
            while ind:
                j = self.edge_to[ind]
                a, b = self.node_scc_id[i], self.node_scc_id[j]
                if a != b:
                    scc_edge.add(a * self.scc_id + b)
                ind = self.edge_next[ind]
        scc_degree = [0] * self.scc_id
        for val in scc_edge:
            scc_degree[val % self.scc_id] += 1
        return scc_edge, scc_degree

    def get_scc_edge_degree_reverse(self):
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
        scc_degree = [0] * self.scc_id
        for val in scc_edge:
            scc_degree[val % self.scc_id] += 1
        return scc_edge, scc_degree, scc_cnt

    def get_scc_node_id(self):
        scc_node_id = [[] for _ in range(self.scc_id)]
        for i in range(self.n):
            scc_node_id[self.node_scc_id[i]].append(i)
        return scc_node_id

    def get_scc_cnt(self):
        scc_cnt = [0] * self.scc_id
        for i in range(self.n):
            scc_cnt[self.node_scc_id[i]] += 1
        return scc_cnt

    def build_new_graph_from_scc_id_to_original_node(self):
        self.initialize_graph()
        for i in range(self.n):
            self.add_directed_edge(self.node_scc_id[i], i)
        return

    def get_original_out_node(self, i):
        ind = self.point_head[i]
        lst = []
        while ind:
            lst.append(self.edge_to[ind])
            ind = self.edge_next[ind]
        return lst


class Tarjan:
    def __init__(self):
        return

    # @staticmethod
    # def get_scc(n: int, edge):
    #     assert all(i not in edge[i] for i in range(n))
    #     assert all(len(set(edge[i])) == len(edge[i]) for i in range(n))
    #     dfs_id = 0
    #     order, low = [math.inf] * n, [math.inf] * n
    #     visit = [0] * n
    #     out = []
    #     in_stack = [0] * n
    #     scc_id = 0
    #     # nodes list of every scc_id part
    #     scc_node_id = []
    #     # index if original node and value is scc_id part
    #     node_scc_id = [-1] * n
    #     parent = [-1] * n
    #     for node in range(n):
    #         if not visit[node]:
    #             stack = [[node, 0]]
    #             while stack:
    #                 cur, ind = stack[-1]
    #                 if not visit[cur]:
    #                     visit[cur] = 1
    #                     order[cur] = low[cur] = dfs_id
    #                     dfs_id += 1
    #                     out.append(cur)
    #                     in_stack[cur] = 1
    #                 if ind == len(edge[cur]):
    #                     stack.pop()
    #                     if order[cur] == low[cur]:
    #                         while out:
    #                             top = out.pop()
    #                             in_stack[top] = 0
    #                             while len(scc_node_id) < scc_id + 1:
    #                                 scc_node_id.append(set())
    #                             scc_node_id[scc_id].add(top)
    #                             node_scc_id[top] = scc_id
    #                             if top == cur:
    #                                 break
    #                         scc_id += 1
    #
    #                     cur, nex = parent[cur], cur
    #                     if cur != -1:
    #                         low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
    #                 else:
    #                     nex = edge[cur][ind]
    #                     stack[-1][-1] += 1
    #                     if not visit[nex]:
    #                         parent[nex] = cur
    #                         stack.append([nex, 0])
    #                     elif in_stack[nex]:
    #                         low[cur] = low[cur] if low[cur] < order[nex] else order[nex]
    #
    #     # new graph after scc
    #     new_dct = [set() for _ in range(scc_id)]
    #     for i in range(n):
    #         for j in edge[i]:
    #             a, b = node_scc_id[i], node_scc_id[j]
    #             if a != b:
    #                 new_dct[a].add(b)
    #     new_degree = [0] * scc_id
    #     for i in range(scc_id):
    #         for j in new_dct[i]:
    #             new_degree[j] += 1
    #     assert len(scc_node_id) == scc_id
    #     return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_pdcc(n: int, edge):

        dfs_id = 0
        order, low = [math.inf] * n, [math.inf] * n
        visit = [False] * n
        out = []
        parent = [-1] * n
        # number of group
        group_id = 0
        # nodes list of every group part
        group_node = []
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
                                    while len(group_node) < group_id + 1:
                                        group_node.append(set())
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

    def get_edcc(self, n: int, edge):
        _, cutting_edges = self.get_cut(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)
        # Remove all cut edges and leaving only edge doubly connected components
        # process the cut edges and then perform bfs on the entire undirected graph
        visit = [0] * n
        edcc_node_id = []
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
            edcc_node_id.append(cur[:])

        # new graph after edcc
        edcc_id = len(edcc_node_id)
        node_edcc_id = [-1] * n
        for i, ls in enumerate(edcc_node_id):
            for x in ls:
                node_edcc_id[x] = i
        new_dct = [[] for _ in range(edcc_id)]
        for i in range(n):
            for j in edge[i]:
                a, b = node_edcc_id[i], node_edcc_id[j]
                if a != b:
                    new_dct[a].append(b)
        new_degree = [0] * edcc_id
        for i in range(edcc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        return edcc_node_id

    @staticmethod
    def get_cut(n: int, edge):
        order, low = [math.inf] * n, [math.inf] * n
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
