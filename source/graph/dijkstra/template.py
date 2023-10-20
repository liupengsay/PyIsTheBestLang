from collections import defaultdict, deque
from heapq import heappush, heappop
from math import inf
from typing import List, Set


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_result(dct: List[List[int]], src: int) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [inf]*n
        stack = [(0, src)]
        dis[src] = 0

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        return dis

    @staticmethod
    def get_dijkstra_cnt(dct: List[List[int]], src: int) -> (List[int], List[any]):
        # 模板: Dijkstra求最短路条数（最短路计算）
        n = len(dct)
        dis = [inf]*n
        stack = [(0, src)]
        dis[src] = 0
        cnt = [0]*n
        cnt[src] = 1
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    # 最短距离更新，重置计数
                    cnt[j] = cnt[i]
                    heappush(stack, (dj, j))
                elif dj == dis[j]:
                    # 最短距离一致，增加计数
                    cnt[j] += cnt[i]
        return cnt, dis

    @staticmethod
    def get_dijkstra_result_limit(dct: List[List[int]], src: int, limit: Set[int], target: Set[int]) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [float("inf")] * n

        dis[src] = 0 if src not in limit else inf
        stack = [(dis[src], src)]
        # 限制只能跑 limit 的点到 target 中的点
        while stack and target:
            d, i = heappop(stack)
            if i in target:
                target.discard(i)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                if j not in limit:
                    dj = w + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))
        return dis

    @staticmethod
    def dijkstra_src_to_dst_path(dct: List[List[int]], src: int, dst: int) -> (List[int], any):
        # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
        n = len(dct)
        dis = [inf] * n
        stack = [(0, src)]
        dis[src] = 0
        father = [-1] * n  # 记录最短路的上一跳
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            if i == dst:
                break
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    father[j] = i
                    heappush(stack, (dj, j))
        if dis[dst] == inf:
            return [], inf
        # 向上回溯路径
        path = []
        i = dst
        while i != -1:
            path.append(i)
            i = father[i]
        return path, dis[dst]

    @staticmethod
    def gen_dijkstra_max_result(dct, src, dsc):

        # 求乘积最大的路，取反后求最短路径
        dis = defaultdict(lambda: float("-inf"))
        stack = [(-1, src)]
        dis[src] = 1
        while stack:
            d, i = heappop(stack)
            d = -d
            if dis[i] > d:
                continue
            for j in dct[i]:
                dj = dct[i][j] * d
                if dj > dis[j]:
                    dis[j] = dj
                    heappush(stack, (-dj, j))
        return dis[dsc]

    @staticmethod
    def get_shortest_by_bfs(dct: List[List[int]], src):
        # 模板: 使用01BFS求最短路
        n = len(dct)
        dis = [-1] * n
        stack = deque([src])
        dis[src] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if dis[j] == -1:
                    dis[j] = dis[i] + 1
                    stack.append(j)
        return dis

    @staticmethod
    def get_second_shortest_path(dct: List[List[int]], src):
        # 模板：使用Dijkstra计算严格次短路  # 也可以计算非严格次短路
        n = len(dct)
        dis = [[inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [(0, src)]
        while stack:
            d, i = heappop(stack)
            if dis[i][1] < d:
                continue
            for j, w in dct[i]:
                if dis[j][0] > d + w:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = d + w
                    heappush(stack, (d + w, j))
                elif dis[j][0] < d + w < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = d + w
                    heappush(stack, (d + w, j))
        return dis

    @staticmethod
    def get_second_shortest_path_cnt(dct: List[List[int]], src, mod=-1):
        # 模板：使用Dijkstra计算严格次短路的条数   # 也可以计算非严格次短路
        n = len(dct)
        dis = [[inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [(0, src, 0)]
        cnt = [[0]*2 for _ in range(n)]
        cnt[src][0] = 1
        while stack:
            d, i, state = heappop(stack)
            if dis[i][1] < d:
                continue
            pre = cnt[i][state]
            for j, w in dct[i]:
                dd = d+w
                if dis[j][0] > dd:
                    dis[j][0] = dd
                    cnt[j][0] = pre
                    heappush(stack, (d + w, j, 0))
                elif dis[j][0] == dd:
                    cnt[j][0] += pre
                    if mod != -1:
                        cnt[j][0] %= mod
                elif dis[j][0] < dd < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = d + w
                    cnt[j][1] = pre
                    heappush(stack, (d + w, j, 1))
                elif dd == dis[j][1]:
                    cnt[j][1] += pre
                    if mod != -1:
                        cnt[j][1] %= mod
        return dis, cnt

    @staticmethod
    def get_shortest_by_bfs_inf_odd(dct: List[List[int]], src):
        # 模板: 使用 01BFS 求最短的奇数距离与偶数距离
        n = len(dct)
        dis = [[inf, inf] for _ in range(n)]
        stack = deque([[src, 0]])
        dis[0][0] = 0
        while stack:
            i, x = stack.popleft()
            for j in dct[i]:
                dd = x + 1
                if dis[j][dd % 2] == inf:
                    dis[j][dd % 2] = x + 1
                    stack.append([j, x + 1])
        return dis

    @staticmethod
    def get_shortest_by_bfs_inf(dct: List[List[int]], src):
        # 模板: 使用 01 BFS 求最短路
        n = len(dct)
        dis = [inf] * n
        stack = deque([src])
        dis[src] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if dis[j] == inf:
                    dis[j] = dis[i] + 1
                    stack.append(j)
        return dis

    @staticmethod
    def get_dijkstra_result_edge(dct: List[List[int]], src: int) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [inf] * n
        stack = [(0, src)]
        dis[src] = 0

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:  # 链式前向星支持自环与重边
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        return dis


class UnDirectedShortestCycle:
    def __init__(self):
        return

    @staticmethod
    def find_shortest_cycle_with_node(n: int, dct) -> int:
        # 模板：求无向图的最小环长度，枚举点
        ans = inf
        for i in range(n):
            dist = [inf] * n
            par = [-1] * n
            dist[i] = 0
            q = [[0, i]]
            while q:
                _, x = heappop(q)
                for child in dct[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] > dct[x][child] + dist[x]:
                        dist[child] = dct[x][child] + dist[x]
                        par[child] = x
                        heappush(q, [dist[child], child])
                    elif par[x] != child and par[child] != x:
                        cur = dist[x] + dist[child] + dct[x][child]
                        ans = ans if ans < cur else cur
        return ans if ans != inf else -1

    @staticmethod
    def find_shortest_cycle_with_edge(n: int, dct, edges) -> int:
        # 模板：求无向图的最小环长度，枚举边

        ans = inf
        for x, y, w in edges:
            dct[x].pop(y)
            dct[y].pop(x)

            dis = [inf] * n
            stack = [[0, x]]
            dis[x] = 0

            while stack:
                d, i = heappop(stack)
                if dis[i] < d:
                    continue
                if i == y:
                    break
                for j in dct[i]:
                    dj = dct[i][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))

            ans = ans if ans < dis[y] + w else dis[y] + w
            dct[x][y] = w
            dct[y][x] = w
        return ans if ans < inf else -1
