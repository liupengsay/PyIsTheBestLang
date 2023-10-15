import math
import unittest
from collections import deque, Counter
from functools import lru_cache
from heapq import nlargest
from itertools import accumulate
from operator import add
from typing import List, Optional

from src.basis.tree_node import TreeNode
from src.data_structure.list_node import ListNode
from src.fast_io import FastIO, inf
from src.graph.union_find import UnionFind



class ReRootDP:
    def __init__(self):
        return 
    
    @staticmethod
    def get_tree_distance_weight(dct: List[List[int]], weight) -> List[int]:
        # 模板：计算树的每个节点到其余所有的节点的总距离（带权重）

        n = len(dct)
        sub = weight[:]  # 子树节点个数
        s = sum(weight)  # 节点的权重值，默认为[1]*n
        ans = [0] * n  # 到其余所有节点的距离之和
        # 第一遍 BFS 自下而上计算子树节点数与 ans[0]
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

        # 第二遍 BFS 自上而下计算距离
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    ans[j] = ans[i] - sub[j] + s - sub[j]
                    stack.append([j, i])
        return ans
    
    @staticmethod
    def get_tree_centroid(dct: List[List[int]]) -> int:
        # 模板：获取树的编号最小的重心
        n = len(dct)
        sub = [1]*n  # 以 0 为根的有根树节点 i 的子树节点数
        ma = [0]*n  # 以 i 为根最大的子树节点数
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
                # 类似换根 DP 计算最大子树节点数
                ma[i] = ma[i] if ma[i] > n-sub[i] else n-sub[i]
                if ma[i] < ma[center] or (ma[i] == ma[center] and i < center):
                    center = i
        # 树的重心等同于最大子树最小的节点也等同于到其余所有节点距离之和最小的节点
        return center

    @staticmethod
    def get_tree_distance(dct: List[List[int]]) -> List[int]:
        # 模板：计算树的每个节点到其余所有的节点的总距离

        n = len(dct)
        sub = [1] * n  # 子树节点个数
        ans = [0] * n  # 到其余所有节点的距离之和

        # 第一遍 BFS 自下而上计算子树节点数与 ans[0]
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

        # 第二遍 BFS 自上而下计算距离
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    ans[j] = ans[i] - sub[j] + n - sub[j]
                    stack.append([j, i])
        return ans

    @staticmethod
    def get_tree_distance_max(dct: List[List[int]]) -> List[int]:
        # 模板：计算树的每个节点到其余所有的节点的最大距离（也可以使用直径上的点BFS）

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # 第一遍 BFS 自下而上计算子树的最大距离与次大距离
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
                        x = sub[j][0]+1
                        if x >= a:
                            a, b = x, a
                        elif x>=b:
                            b = x
                sub[i] = [a, b]

        # 第二遍 BFS 自上而下更新最大距离
        stack = [[0, -1, 0]]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0]+1
                    a, b = sub[i]
                    # 排除当前子节点的距离
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append([j, i, nex+1])
        return ans
    
    
class TreeDiameterWeighted:
    def __init__(self):
        return

    @staticmethod
    def bfs(dct, src):
        # 模板：使用 BFS 计算获取带权树的直径端点以及直径长度
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
            # 模板：使用BFS计算获取树的直径端点以及直径长度
            d = 0
            q = deque([(node, -1, d)])
            while q:
                node, pre, d = q.popleft()
                for nex in edge[node]:
                    if nex != pre:
                        q.append((nex, node, d + 1))
            return node, d

        n = len(edge)

        # 这个算法依赖于一个性质，对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点
        x, _ = bfs(0)
        # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径
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

        # 模板：使用DFS与动态规划计算直径
        ans = 0
        dfs(0, -1)
        return ans


class TreeDiameterInfo:
    def __init__(self):
        return

    @staticmethod
    def get_diameter_info(edge: List[List[int]], root=0):
        # 模板：使用两遍 BFS 计算获取不带权的树直径端点以及直径长度和具体直径经过的点
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
    # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径。
    # 这个算法依赖于一个性质：对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点。
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
        # 获取树的直径端点
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
    

