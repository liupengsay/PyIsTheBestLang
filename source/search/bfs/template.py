
from collections import deque
from math import inf
from typing import List, Dict

from graph.dijkstra.template import Dijkstra


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


class TreeDiameterWeight:
    def __init__(self):
        return

    @staticmethod
    def get_diameter_with_dijkstra(dct):
        """template of find the diameter with weighted undirected edge"""
        dis = Dijkstra().get_shortest_path(dct, 0)
        x = dis.index(max(dis))

        dis = Dijkstra().get_shortest_path(dct, x)
        y = dis.index(max(dis))
        path, dis = Dijkstra().get_shortest_path_from_src_to_dst(dct, x, y)
        return path, dis
