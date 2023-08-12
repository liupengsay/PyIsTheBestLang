import sys
from collections import defaultdict
from collections import deque
from types import GeneratorType


class FastIO:
    def __init__(self):
        self.inf = float("inf")
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return float(self._read())

    def read_ints(self):
        return map(int, self._read().split())

    def read_floats(self):
        return map(float, self._read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self._read().split())

    def read_list_ints(self):
        return list(map(int, self._read().split()))

    def read_list_floats(self):
        return list(map(float, self._read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self._read().split()))

    def read_str(self):
        return self._read()

    def read_list_strs(self):
        return self._read().split()

    def read_list_str(self):
        return list(self._read())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def bootstrap(f, queue=deque()):
        def wrappedfunc(*args, **kwargs):
            if queue:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        queue.append(to)
                        to = next(to)
                    else:
                        queue.pop()
                        if not queue:
                            break
                        to = queue[-1].send(to)
                return to

        return wrappedfunc


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
        edge = [[0]*self.n for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] += 1
            edge[i][j] += 1
            edge[j][i] += 1

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

        """
        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            for nex in range(self.n):
                while edge[pre][nex]:
                    edge[pre][nex] -= 1
                    edge[nex][pre] -= 1
                    dfs(nex)
                    self.nodes.append(nex)
                    self.paths.append([pre, nex])
            return

        dfs(starts[0])
        self.paths.reverse()
        self.nodes.append(starts[0])
        self.nodes.reverse()
        """
        # 使用迭代版本Hierholzer算法计算
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            next_node = None
            for nex in range(self.n):
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


# 标准并查集
class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


def main(ac=FastIO()):
    m = ac.read_int()
    nodes = set()
    pairs = []
    for _ in range(m):
        s = ac.read_str()
        nodes.add(s[0])
        nodes.add(s[1])
        pairs.append([s[0], s[1]])
    nodes = sorted(list(nodes))
    ind = {num: i for i, num in enumerate(nodes)}
    n = len(nodes)
    uf = UnionFind(n)
    for x, y in pairs:
        uf.union(ind[x], ind[y])
    if uf.part != 1:
        ac.st("No Solution")
        return
    pairs = [[ind[x], ind[y]] for x, y in pairs]
    euler = UnDirectedEulerPath(n, pairs)
    if not euler.exist:
        ac.st("No Solution")
        return
    ans = "".join([nodes[a] for a in euler.nodes])
    ac.st(ans)
    return


main()
