from collections import deque
from math import inf
from typing import List, Dict


class ReRootDP:
    def __init__(self):
        return

    @staticmethod
    def get_tree_distance_weight(dct: List[List[int]], weight) -> List[int]:
        # Calculate the total distance from each node of the tree to all other nodes
        # each node has weight

        n = len(dct)
        sub = weight[:]
        s = sum(weight)  # default equal to [1]*n
        ans = [0] * n  # distance to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + s - sub[j]
                    stack.append([j, i])
        return ans

    @staticmethod
    def get_tree_centroid(dct: List[List[int]]) -> int:
        # the smallest centroid of tree
        # equal the node with minimum of maximum subtree node cnt
        # equivalent to the node which has the shortest distance from all other nodes
        n = len(dct)
        sub = [1] * n  # subtree size of i-th node rooted by 0
        ma = [0] * n  # maximum subtree node cnt or i-rooted
        ma[0] = n
        center = 0
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ma[i] = ma[i] if ma[i] > sub[j] else sub[j]
                # like re-rooted dp to check the maximum subtree size
                ma[i] = ma[i] if ma[i] > n - sub[i] else n - sub[i]
                if ma[i] < ma[center] or (ma[i] == ma[center] and i < center):
                    center = i
        return center

    @staticmethod
    def get_tree_distance(dct: List[List[int]]) -> List[int]:
        # Calculate the total distance from each node of the tree to all other nodes

        n = len(dct)
        sub = [1] * n  # Number of subtree nodes
        ans = [0] * n  # The sum of distances to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [(0, -1, 1)]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append((i, fa, 0))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i, 1))
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + n - sub[j]
                    stack.append((j, i))
        return ans

    @staticmethod
    def get_tree_distance_max(dct: List[List[int]]) -> List[int]:
        # Calculate the maximum distance from each node of the tree to all other nodes
        # point BFS on diameter can also be used

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # first BFS compute the largest distance and second large distance from bottom to up
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0] + 1
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                sub[i] = [a, b]

        # second BFS compute large distance from up to bottom
        stack = [(0, -1, 0)]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0] + 1
                    a, b = sub[i]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append((j, i, nex + 1))
        return ans


class TreeDiameterWeighted:
    def __init__(self):
        return

    @staticmethod
    def bfs(dct: List[Dict[int]], src: int) -> (int, List[int], int):
        # Using BFS calculation to obtain the diameter endpoint and diameter length of a weighted tree
        n = len(dct)
        res = [inf] * n
        stack = [src]
        res[src] = 0
        parent = [-1] * n
        while stack:
            node = stack.pop()
            for nex in dct[node]:
                if nex != parent[node]:
                    parent[nex] = node
                    res[nex] = res[node] + dct[node][nex]
                    stack.append(nex)
        far = res.index(max(res))
        diameter = [far]
        while diameter[-1] != src:
            diameter.append(parent[diameter[-1]])
        diameter.reverse()
        return far, diameter, res[far]


class TreeDiameter:
    def __init__(self):
        return

    @staticmethod
    def get_diameter_bfs(edge):

        def bfs(node):
            # Use BFS calculation to obtain the diameter endpoint and diameter length of the tree
            d = 0
            q = deque([(node, -1, d)])
            while q:
                node, pre, d = q.popleft()
                for nex in edge[node]:
                    if nex != pre:
                        q.append((nex, node, d + 1))
            return node, d

        # This algorithm relies on a property that for any point in the tree
        # the farthest point from it must be an endpoint of a diameter on the tree
        x, _ = bfs(0)
        # Take any node x in the tree
        # find the farthest point y from it
        # and point y is the endpoint of a diameter in the tree
        # We start from y and find the point farthest from y to find a diameter
        y, dis = bfs(x)
        return dis

    @staticmethod
    def get_diameter_dfs(edge):

        def dfs(i, fa):
            nonlocal ans
            a = b = 0
            for j in edge[i]:
                if j != fa:
                    x = dfs(j, i)
                    if x >= a:
                        a, b = x, a
                    elif x >= b:
                        b = x
            ans = ans if ans > a + b else a + b
            return a + 1 if a > b else b + 1

        # Calculate diameter using DFS and dynamic programming
        ans = 0
        dfs(0, -1)
        return ans


class TreeDiameterInfo:
    def __init__(self):
        return

    @staticmethod
    def get_diameter_info(edge: List[List[int]], root=0):
        # Use two rounds of BFS calculation to obtain the endpoint of the tree diameter without weight
        # as well as the diameter length and the specific point through which the diameter passes
        n = len(edge)

        stack = deque([[root, -1]])
        parent = [-1] * n
        dis = [0] * n
        x = -1
        while stack:
            i, fa = stack.popleft()
            x = i
            for j in edge[i]:
                if j != fa:
                    parent[j] = i
                    dis[j] = dis[i] + 1
                    stack.append([j, i])

        stack = deque([[x, -1]])
        parent = [-1] * n
        dis = [0] * n
        y = -1
        while stack:
            i, fa = stack.popleft()
            y = i
            for j in edge[i]:
                if j != fa:
                    parent[j] = i
                    dis[j] = dis[i] + 1
                    stack.append([j, i])

        path = [y]
        while path[-1] != x:
            path.append(parent[path[-1]])
        return x, y, dis, path


class TreeDiameterDis:

    def __init__(self, edge):
        self.edge = edge
        self.n = len(self.edge)
        return

    def get_furthest(self, node):
        q = deque([(node, -1)])
        while q:
            node, pre = q.popleft()
            for x in self.edge[node]:
                if x != pre:
                    q.append((x, node))
        return node

    def get_diameter_node(self):
        x = self.get_furthest(0)
        y = self.get_furthest(x)
        return x, y

    def get_bfs_dis(self, node):
        dis = [inf] * self.n
        stack = [node]
        dis[node] = 0
        while stack:
            nex = []
            for i in stack:
                for j in self.edge[i]:
                    if dis[j] == inf:
                        nex.append(j)
                        dis[j] = dis[i] + 1
            stack = nex[:]
        return dis
