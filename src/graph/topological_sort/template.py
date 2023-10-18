class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1] * n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    @staticmethod
    def count_dag_path(n, edges):
        # 模板: 计算有向无环连通图路径条数
        edge = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:  # 也可以使用深搜
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return cnt

    # 内向基环树写法 https://atcoder.jp/contests/abc266/submissions/37717739
    # def main(ac=FastIO()):
    #     n = ac.read_int()
    #     edge = [[] for _ in range(n)]
    #     uf = UnionFind(n)
    #
    #     degree = [0] * n
    #     for _ in range(n):
    #         u, v = ac.read_list_ints_minus_one()
    #         edge[u].append(v)
    #         edge[v].append(u)
    #         degree[u] += 1
    #         degree[v] += 1
    #
    #     que = deque()
    #     for i in range(n):
    #         if degree[i] == 1:
    #             que.append(i)
    #     while que:
    #         now = que.popleft()
    #         nex = edge[now][0]
    #         degree[now] -= 1
    #         degree[nex] -= 1
    #         edge[nex].remove(now)
    #         uf.union(now, nex)
    #         if degree[nex] == 1:
    #             que.append(nex)
    #
    #     q = ac.read_int()
    #     for _ in range(q):
    #         x, y = ac.read_list_ints_minus_one()
    #         if uf.is_connected(x, y):
    #             ac.st("Yes")
    #         else:
    #             ac.st("No")
    #     return

    @staticmethod
    def is_topology_unique(dct, degree, n):

        # 保证存在拓扑排序的情况下判断是否唯一
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):

        # 使用拓扑排序判断有向图是否存在环
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return all(x == 0 for x in degree)

    @staticmethod
    def bfs_topologic_order(n, dct, degree):
        # 拓扑排序判断有向图是否存在环，同时记录节点的拓扑顺序
        order = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        ind = 0
        while stack:
            nex = []
            for i in stack:
                order[i] = ind
                ind += 1
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            return []
        return order


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
