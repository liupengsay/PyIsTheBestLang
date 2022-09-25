class Solution:
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        dct = defaultdict(list)
        for a, b in edges:
            dct[a].append(b)
            dct[b].append(a)
        ind = defaultdict(list)
        for i in range(n):
            ind[vals[i]].append(i)

        def dfs(i):
            nonlocal ans
            if visit[i]:
                return
            if vals[i] == k and i!=x:
                ans += 1
            visit[i] = 1
            for j in dct[i]:
                if vals[j] <= vals[i] and not visit[j]:
                    dfs(j)
            return

        ans = n
        for k in ind:
            i = ind[k][0]
            stack = [[k, i]]
            visit = [0]*n
            visit[i] = 1
            while stack:
                nex = []
                for path in stack:
                    if vals[path[-1]] == k and len(path)>2:
                        ans += 1
                    for j in dct[path[-1]]:
                        if not visit[j] and vals[j] <= path[0]:
                            nex.append(path+[j])
                            visit[j] = 1
                stack = nex[:]
        return ans