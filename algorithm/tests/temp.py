
# 标准并查集
class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n

    def find(self, x):
        if x != self.root[x]:
            # 在查询的时候合并到顺带直接根节点
            root_x = self.find(self.root[x])
            self.root[x] = root_x
            return root_x
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



class Solution:
    def isPossibleToCutPath(self, grid: List[List[int]]) -> bool:
        m, n = len(grid), len(grid[0])
        # 建图
        dp = [[0 ] *n for _ in range(m)]
        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    for a, b in [[ i -1, j], [i, j- 1]]:
                        if 0 <= a < m and 0 <= b < n and dp[a][b]:
                            dp[i][j] = 1

        post = [[0] * n for _ in range(m)]
        post[m - 1][n - 1] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if grid[i][j] == 1:
                    for a, b in [[i + 1, j], [i, j + 1]]:
                        if 0 <= a < m and 0 <= b < n and post[a][b]:
                            post[i][j] = 1

        cnt = 0
        flag = False
        for i in range(m):
            for j in range(n):
                if [i, j] != [0, 0] and [i, j] != [m - 1, n - 1] and grid[i][j]:
                    left = 0
                    for a, b in [[i - 1, j], [i, j - 1]]:
                        if 0 <= a < m and 0 <= b < n and dp[a][b] and grid[a][b]:
                            left = 1

                    right = 0
                    for a, b in [[i + 1, j], [i, j + 1]]:
                        if 0 <= a < m and 0 <= b < n and post[a][b] and grid[a][b]:
                            right = 1

                    if left and right:
                        continue
                    else:
                        grid[i][j] = 0

        m, n = len(grid), len(grid[0])
        # 建图
        edge = defaultdict(list)
        uf = UnionFind(m * n)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if i + 1 < m and grid[i + 1][j] == 1:
                        edge[i * n + j].append(i * n + n + j)
                        edge[i * n + n + j].append(i * n + j)
                        uf.union(i * n + j, i * n + n + j)
                    if j + 1 < n and grid[i][j + 1] == 1:
                        edge[i * n + j].append(i * n + 1 + j)
                        edge[i * n + 1 + j].append(i * n + j)
                        uf.union(i * n + j, i * n + 1 + j)
                    if not edge[i * n + j]:
                        edge[i * n + j] = []

        if not uf.is_connected(0, m * n - 1):
            return True

        # 找寻割点数目
        visit = dict()
        root = dict()
        index = 1
        ans = set()
        stack = []

        def tarjan(i, parent):
            nonlocal index, stack
            visit[i] = root[i] = index
            index += 1
            stack.append(i)
            child = 0
            for j in edge[i]:
                if j == parent:
                    continue
                elif j not in visit:
                    child += 1
                    tarjan(j, i)
                    root[i] = min(root[i], root[j])
                    # 两种情况下才为割点
                    if parent != -1 and visit[i] <= root[j]:
                        ans.add(i)
                    elif parent == -1 and child >= 2:
                        ans.add(i)
                else:
                    root[i] = min(root[i], visit[j])
            if visit[i] == root[i]:
                lst = []
                while stack[-1] != i:
                    lst.append(stack.pop(-1))
                lst.append(stack.pop(-1))
                for ls in lst:
                    root[ls] = root[i]
            return

        tarjan(list(edge.keys())[0], -1)
        ans.discard(0)
        ans.discard(m * n - 1)
        return len(ans) > 0
