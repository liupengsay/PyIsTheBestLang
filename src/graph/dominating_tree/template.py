

# 定义一个支配树类
class DominatingTree:
    def __init__(self, n):
        # 初始化支配树，n 为图中点的数量
        self.n = n
        self.edges = []  # 存储图中的边
        self.dominators = [-1] * n  # 存储每个点的支配点

    def add_edge(self, u, v):
        # 添加一条从点 u 到点 v 的边
        self.edges.append((u, v))

    def build(self):
        # 构建支配树
        # 使用并查集维护每个点的支配点
        parent = [i for i in range(self.n)]

        # 遍历图中的每条边，更新支配点
        for (u, v) in self.edges:
            pu = self.find(parent, u)
            pv = self.find(parent, v)
            if pu != pv:
                parent[pv] = pu

        # 所有节点的支配点都是根节点
        for i in range(self.n):
            self.dominators[i] = self.find(parent, i)

    def find(self, parent, i):
        # 并查集的 find 操作
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def get_dominators(self):
        # 返回每个点的支配点
        return self.dominators

