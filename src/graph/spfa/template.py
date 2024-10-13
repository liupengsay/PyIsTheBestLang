from collections import deque




class SPFA:
    def __init__(self):
        return

    @staticmethod
    def negative_circle_edge(dct, src=0, initial=0):
        """
        determine whether there is a negative loop and find the shortest path
        """
        # Finding the shortest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [math.inf] * n
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
                        return True, dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return False, dis, cnt

    @staticmethod
    def positive_circle_edge(dct, src=0, initial=0):
        """
        determine whether there is a positive loop and find the longest path
        """
        # Finding the longest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [-math.inf] * n
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
                        return True, dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return False, dis, cnt

    @staticmethod
    def negative_circle_mul(dct, src=0, initial=0):
        """Determine if there is a ring with a product greater than 1"""
        n = len(dct)
        dis = [math.inf for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:
                if dis[v] > dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return True, dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return False, dis, cnt

    @staticmethod
    def positive_circle_mul(dct, src=0, initial=1):
        """Determine if there is a ring with a product greater than 1"""
        n = len(dct)
        dis = [0 for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:
                if dis[v] < dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return True, dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return False, dis, cnt
