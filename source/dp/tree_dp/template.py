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
