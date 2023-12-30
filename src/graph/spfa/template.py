from collections import deque

from src.utils.fast_io import inf


class SPFA:
    def __init__(self):
        return

    @staticmethod
    def negative_circle_edge(dct, src=0, initial=0):
        """determine whether there is a negative loop and find the shortest path
        which can also find a positive loop by make the opposite weight of the graph
        """
        # Finding the shortest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [inf] * n
        # flag of node in stack or not
        visit = [False] * n
        # the number of edges by the shortest path
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True
        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:  # Chain forward stars support self loops and double edges
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        # there is at least one negative loop starting from the starting point
                        return "YES", dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return "NO", dis, cnt

    @staticmethod
    def positive_circle_edge(dct, src=0, initial=0):
        """determine whether there is a negative loop and find the shortest path
        which can also find a positive loop by make the opposite weight of the graph
        """
        # Finding the shortest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [-inf] * n
        # flag of node in stack or not
        visit = [False] * n
        # the number of edges by the shortest path
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True
        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:  # Chain forward stars support self loops and double edges
                if dis[v] < dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        # there is at least one negative loop starting from the starting point
                        return "YES", dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return "NO", dis, cnt

    @staticmethod
    def negative_circle(dct, src=0, initial=0):
        n = len(dct)
        dis = [inf for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return "YES", dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return "NO", dis, cnt

    @staticmethod
    def count_shortest_path(dct, mod=10 ** 9 + 7):
        # The Shortest Path Count of Undirected Unauthorized Graphs
        n = len(dct)
        dis = [inf for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([0])
        dis[0] = 0
        visit[0] = True
        cnt[0] = 1
        while queue:
            u = queue.popleft()
            visit[u] = False
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] + 1:
                    dis[v] = dis[u] + 1
                    cnt[v] = w * cnt[u]
                    cnt[v] %= mod
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
                elif dis[v] == dis[u] + 1:
                    cnt[v] += w * cnt[u]
                    cnt[v] %= mod
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return cnt

    @staticmethod
    def negative_circle_mul(dct, src=0, initial=0):
        """Determine if there is a ring with a product greater than 1"""
        n = len(dct)
        dis = [inf for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return "YES", dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return "NO", dis, cnt

    def differential_constraint(self, ineq, n: int):
        """find is there a solution to the inequality system of differential constraint calculation"""
        dct = [dict() for _ in range(n + 1)]
        # original node index start at 1
        # virtual node 0 as root
        for i in range(1, n + 1):
            dct[0][i] = 0
        for a, b, c in ineq:  # a-b<=c
            w = dct[b].get(a, inf)  # constraints with smaller values
            w = w if w < c else c
            dct[b][a] = w
        # use the shortest path and negative circle to judge if there has a circle
        ans, dis, _ = self.negative_circle(dct, 0, 0)
        return ans, dis
