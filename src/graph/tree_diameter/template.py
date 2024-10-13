


class GraphDiameter:
    def __init__(self):
        return

    @staticmethod
    def get_diameter(dct, root=0):
        n = len(dct)
        dis = [math.inf] * n
        stack = [root]
        dis[root] = 0
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex[:]
        root = dis.index(max(dis))
        dis = [math.inf] * n
        stack = [root]
        dis[root] = 0
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex[:]
        return max(dis)


class TreeDiameter:
    def __init__(self, dct):
        self.n = len(dct)
        self.dct = dct
        return

    def get_bfs_dis(self, root):
        dis = [math.inf] * self.n
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

    def get_diameter_math.info(self):
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
