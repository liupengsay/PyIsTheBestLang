from collections import deque
from typing import List


class DirectedEulerPath:
    def __init__(self, n, pairs: List[List[int]]):
        self.n = n
        # directed edge
        self.pairs = pairs
        # edges order on euler path
        self.paths = list()
        # nodes order on euler path
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # in and out degree sum of node
        degree = [0]*self.n
        edge = [[] for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] -= 1
            edge[i].append(j)

        # visited by lexicographical order
        for i in range(self.n):
            edge[i].sort(reverse=True)  # which can be adjusted

        # find the start point and end point of euler path
        starts = []
        ends = []
        zero = 0
        for i in range(self.n):
            if degree[i] == 1:
                starts.append(i)  # start node which out_degree - in_degree = 1
            elif degree[i] == -1:
                ends.append(i)  # start node which out_degree - in_degree = -1
            else:
                zero += 1  # other nodes have out_degree - in_degree = 0
        del degree

        if not len(starts) == len(ends) == 1:
            if zero != self.n:
                return
            starts = [0]

        # Hierholzer algorithm with iterative implementation
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            if edge[current]:
                next_node = edge[current].pop()
                stack.append(next_node)
            else:
                self.nodes.append(current)
                if len(stack) > 1:
                    self.paths.append([stack[-2], current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()

        # Pay attention to determining which edge passes through before calculating the Euler path
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class UnDirectedEulerPath:
    def __init__(self, n, pairs: List[int]):
        self.n = n
        # undirected edge
        self.pairs = pairs
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        degree = [0]*self.n
        edge = [dict() for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] += 1
            edge[i][j] = edge[i].get(j, 0) + 1
            edge[j][i] = edge[j].get(i, 0) + 1
        edge_dct = [deque(sorted(dt)) for dt in edge]  # visited by order of node id
        starts = []
        zero = 0
        for i in range(self.n):
            if degree[i] % 2:  # which can be start point or end point
                starts.append(i)
            else:
                zero += 1
        del degree

        if not len(starts) == 2:
            # just two nodes have odd degree and others have even degree
            if zero != self.n:
                return
            starts = [0]

        # Hierholzer algorithm with iterative implementation
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            next_node = None
            while edge_dct[current]:
                if not edge[current][edge_dct[current][0]]:
                    edge_dct[current].popleft()
                    continue
                nex = edge_dct[current][0]
                if edge[current][nex]:
                    edge[current][nex] -= 1
                    edge[nex][current] -= 1
                    next_node = nex
                    stack.append(next_node)
                    break
            if next_node is None:
                self.nodes.append(current)
                if len(stack) > 1:
                    pre = stack[-2]
                    self.paths.append([pre, current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()
        # Pay attention to determining which edge passes through before calculating the Euler path
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return
