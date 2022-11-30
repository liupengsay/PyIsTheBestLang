

import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, permutations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from functools import lru_cache


from functools import lru_cache

n, m = [int(w) for w in input().strip().split() if w]
weight = [int(w) for w in input().strip().split() if w]
edges =  [[] for _ in range(n)]
for _ in range(m):
    u, v = [int(w) for w in input().strip().split() if w]
    edges[u-1].append(v-1)


def check_graph(edge: List[list], n):
    """

    :param edge: 边连接关系
    :param n: 节点数
    :return:
    """
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
            if not visit[j]:
                child += 1
                tarjan(j, i)
                root[i] = min(root[i], root[j])
            elif in_stack[j]:
                root[i] = min(root[i], visit[j])

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
        return

    for k in range(n):
        if not visit[k]:
            tarjan(k, -1)
    return sub_group

#print(edges)
# 有向有环图
sub_group = check_graph(edges, n)
#print(sub_group)
k = len(sub_group)
dct = dict()
for i in range(k):
    for j in sub_group[i]:
        dct[j] = i
new_weight = [sum(weight[j] for j in sub) for sub in sub_group]
new_edges = [[] for _ in range(k)]
for i in range(n):
    for j in edges[i]:
        if dct[i] != dct[j]:
            new_edges[dct[i]].append(dct[j])
#print(new_edges)
@lru_cache(None)
def dfs(x):
    res = new_weight[x]
    add = 0
    for y in new_edges[x]:
        add = add if add > dfs(y) else dfs(y)
    return res + add

ans = max(dfs(i) for i in range(k))
dfs.cache_clear()
print(ans)