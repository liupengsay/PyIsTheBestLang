

""""
SPFA：路径边数优先的广度优先搜索（使用带负权值）
P3385 【模板】负环
SPFA（Shortest Path Faster Algorithm）是一种用于计算单源最短路径的算法。它通过使用队列和松弛操作来不断更新路径长度，从而更快地找到最短路径。

下面是一个简单的 Python SPFA 模板，其中 graph 是图的邻接表表示，src 是源节点，dist 是各节点到源节点的最短距离，prev 是各节点的前驱节点。
上面的代码只是一个简单的 SPFA 模板，实际使用时可能需要添加更多的特判和优化。例如，SPFA 算法在某些情况下容易陷入死循环，因此需要添加防止死循环的机制。此外，SPFA 算法的时间复杂度与输入图

的稠密程度有关，因此可能需要使用一些优化方法来提高它的效率。

总之，SPFA 算法是一种简单易用的最短路径算法，它通过使用队列和松弛操作来快速求解单源最短路径问题。它的时间复杂度与输入图的稠密程度有关，并且容易陷入死循环，因此需要注意这些问题。


Dijkstra：路径权值优先的深度优先搜索（只适用正权值）
"""
import heapq
import sys
from collections import defaultdict, Counter, deque
from functools import lru_cache
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)

# def dijkstra():
#     distance = [float("inf") for _ in range(n + 1)]
#     visit = [float("inf")] * (n + 1)
#     stack = [[0, 1, 0]]
#     visit[1] = 0
#     while stack:
#         dis, u, cnt = heapq.heappop(stack)
#         if distance[u] <= dis:
#             continue
#         if visit[u] < cnt:
#             return "YES"
#         visit[u] = cnt
#         distance[u] = dis
#         for v, w in dct[u].items():
#             heapq.heappush(stack, [dis+w, v, cnt+1])
#     return "NO"

# def dijkstra():
#     distance = [float("inf") for _ in range(n + 1)]
#     stack = [[0, 0, 1]]
#     while stack:
#         cnt, dis, u = heapq.heappop(stack)
#         if distance[u] <= dis:
#             continue
#         if cnt >= n:
#             return "Yes"
#         distance[u] = dis
#         for v, w in dct[u]:
#             heapq.heappush(stack, [cnt+1, dis+w])
#     return "NO"


# # 初始化距离和前驱
# dist = [float('inf')] * len(graph)
# prev = [None] * len(graph)
# dist[src] = 0
#
# # 创建一个队列用于存储节点
# queue = []
# queue.append(src)
#
# while queue:
#     # 取出队列中的第一个节点
#     curr = queue.pop(0)
#
#     # 更新当前节点的相邻节点的距离
#     for neighbor, weight in graph[curr]:
#         if dist[neighbor] > dist[curr] + weight:
#             dist[neighbor] = dist[curr] + weight
#             prev[neighbor] = curr
#
#             # 如果相邻节点还没有在队列中，将它加入队列
#             if neighbor not in queue:
#                 queue.append(neighbor)


def spfa():
    # 求带负权的最短路距离与路径边数
    # 距离
    dis = [float("inf") for _ in range(n + 1)]
    # 是否在栈中
    visit = [False] * (n + 1)
    # 当前最小距离的路径边数
    cnt = [0] * (n + 1)

    # 队列与点1初始化
    queue = deque([1])
    dis[1] = 0
    visit[1] = True

    while queue:
        # 出队
        u = queue.popleft()
        visit[u] = False
        for v, w in dct[u].items():
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                cnt[v] = cnt[u] + 1
                if cnt[v] >= n:
                    return "YES"
                if not visit[v]:
                    queue.append(v)
                    visit[v] = True
    return "NO"

t = int(input().strip())
for _ in range(t):
    n, m = [int(w) for w in input().split() if w]
    dct = [defaultdict(lambda: float("inf")) for _ in range(n+1)]
    # 注意有很多重边
    for _ in range(m):
        u, v, w = [int(w) for w in input().split() if w]
        dct[u][v] = min(dct[u][v], w)
        if w >= 0:
            dct[v][u] = min(dct[v][u], w)
    print(spfa())
