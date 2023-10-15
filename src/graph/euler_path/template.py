


class DirectedEulerPath:
    def __init__(self, n, pairs):
        # 数组形式存储的有向连接关系
        self.n = n
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = [0]*self.n
        # 存储图关系
        edge = [[] for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] -= 1
            edge[i].append(j)

        # 根据字典序优先访问较小的
        for i in range(self.n):
            edge[i].sort(reverse=True)

        # 寻找起始节点
        starts = []
        ends = []
        zero = 0
        for i in range(self.n):
            if degree[i] == 1:
                starts.append(i)
            elif degree[i] == -1:
                ends.append(i)
            else:
                zero += 1
        del degree

        # 图中恰好存在 1 个点出度比入度多 1（这个点即为起点） 1 个点出度比入度少 1（这个点即为终点）其余相等
        if not len(starts) == len(ends) == 1:
            if zero != self.n:
                return
            starts = [0]

        # 使用迭代版本的Hierholzer算法
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            if edge[current]:
                next_node = edge[current].pop()
                stack.append(next_node)
            else:
                self.nodes.append(current)
                if len(stack) > 1:
                    self.paths.append([stack[-2], current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()
        """
        改进前的深搜版本
        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            while edge[pre]:
                nex = edge[pre].pop()
                dfs(nex)
                self.nodes.append(nex)
                self.paths.append([pre, nex])
            return

        dfs(starts[0])
        self.paths.reverse()
        self.nodes.append(starts[0])
        self.nodes.reverse()
        """
        # 注意判断所有边都经过的才算欧拉路径
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class UnDirectedEulerPath:
    def __init__(self, n, pairs):
        # 数组形式存储的有向连接关系
        self.n = n
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = [0]*self.n
        # 存储图关系
        edge = [dict() for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] += 1
            edge[i][j] = edge[i].get(j, 0) + 1
            edge[j][i] = edge[j].get(i, 0) + 1
        edge_dct = [deque(sorted(dt)) for dt in edge]  # 从小到大进行序号遍历
        # 寻找起始节点
        starts = []
        zero = 0
        for i in range(self.n):
            # 如果有节点出度比入度恰好多 1，那么只有它才能是起始
            if degree[i] % 2:
                starts.append(i)
            else:
                zero += 1
        del degree

        # 不存在欧拉路径
        if not len(starts) == 2:
            if zero != self.n:
                return
            starts = [0]

        # 使用迭代版本Hierholzer算法计算
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            next_node = None
            while edge_dct[current]:
                if not edge[current][edge_dct[current][0]]:
                    edge_dct[current].popleft()
                    continue
                nex = edge_dct[current][0]
                if edge[current][nex]:
                    edge[current][nex] -= 1
                    edge[nex][current] -= 1
                    next_node = nex
                    stack.append(next_node)
                    break
            if next_node is None:
                self.nodes.append(current)
                if len(stack) > 1:
                    pre = stack[-2]
                    self.paths.append([pre, current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()
        # 注意判断所有边都经过的才算欧拉路径
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return

