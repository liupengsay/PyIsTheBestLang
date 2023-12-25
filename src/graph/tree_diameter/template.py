from typing import List

from src.utils.fast_io import inf


class TreeDiameter:
    def __init__(self, dct: List[List[List[int]]]):
        self.n = len(dct)
        self.dct = dct
        return

    def get_bfs_dis(self, root) -> (List[any], List[int]):
        dis = [inf] * self.n
        stack = [root]
        dis[root] = 0
        parent = [-1] * self.n
        while stack:
            i = stack.pop()
            for j, w in self.dct[i]:  # weighted edge
                if j != parent[i]:
                    parent[j] = i
                    dis[j] = dis[i] + w
                    stack.append(j)
        return dis, parent

    def get_diameter_info(self) -> (int, int, List[int], any):
        """get tree diameter detail by weighted bfs twice"""
        dis, _ = self.get_bfs_dis(0)
        x = dis.index(max(dis))
        dis, parent = self.get_bfs_dis(x)
        y = dis.index(max(dis))
        path = [y]
        while path[-1] != x:
            path.append(parent[path[-1]])
        path.reverse()
        return x, y, path, dis[y]
