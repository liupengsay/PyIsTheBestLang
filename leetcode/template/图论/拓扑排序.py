class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        dct = defaultdict(list)
        degree = defaultdict(int)
        for i, j in relations:
            degree[j] += 1
            dct[i].append(j)
        cnt = 0
        step = 0
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            cnt += len(stack)
            step += 1
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return step if cnt == n else -1