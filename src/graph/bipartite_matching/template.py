class BipartiteMatching:
    def __init__(self, n, m):
        self._n = n
        self._m = m
        self._to = [[] for _ in range(n)]

    def add_edge(self, a, b):
        self._to[a].append(b)

    def solve(self):
        n, m, to = self._n, self._m, self._to
        prev = [-1] * n
        root = [-1] * n
        p = [-1] * n
        q = [-1] * m
        updated = True
        while updated:
            updated = False
            s = []
            s_front = 0
            for i in range(n):
                if p[i] == -1:
                    root[i] = i
                    s.append(i)
            while s_front < len(s):
                v = s[s_front]
                s_front += 1
                if p[root[v]] != -1:
                    continue
                for u in to[v]:
                    if q[u] == -1:
                        while u != -1:
                            q[u] = v
                            p[v], u = u, p[v]
                            v = prev[v]
                        updated = True
                        break
                    u = q[u]
                    if prev[u] != -1:
                        continue
                    prev[u] = v
                    root[u] = root[v]
                    s.append(u)
            if updated:
                for i in range(n):
                    prev[i] = -1
                    root[i] = -1
        return [(v, p[v]) for v in range(n) if p[v] != -1]


class Hungarian:
    def __init__(self):
        # Bipartite graph maximum math without weight
        return

    @staticmethod
    def dfs_recursion(n, m, dct):
        assert len(dct) == m

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        # left group size is n
        match = [-1] * n
        ans = 0
        for x in range(m):
            # right group size is m
            visit = [False] * n
            if hungarian(x):
                ans += 1
        return ans

    @staticmethod
    def bfs_iteration(n, m, dct):

        assert len(dct) == m

        match = [-1] * n
        ans = 0
        for i in range(m):
            hungarian = [0] * m
            visit = [0] * n
            stack = [[i, 0]]
            while stack:
                x, ind = stack[-1]
                if ind == len(dct[x]) or hungarian[x]:
                    stack.pop()
                    continue
                y = dct[x][ind]
                if not visit[y]:
                    visit[y] = 1
                    if match[y] == -1:
                        match[y] = x
                        hungarian[x] = 1
                    else:
                        stack.append([match[y], 0])
                else:
                    if hungarian[match[y]]:
                        match[y] = x
                        hungarian[x] = 1
                    stack[-1][1] += 1
            if hungarian[i]:
                ans += 1
        return ans
