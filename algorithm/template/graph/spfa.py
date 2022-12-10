

""""
SPFA：路径边数优先的广度优先搜索（使用带负权值）
P3385 【模板】负环

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
