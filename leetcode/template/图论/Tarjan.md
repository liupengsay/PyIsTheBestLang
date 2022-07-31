import heapq

class Solution:
    def distanceToCycle(self, n, edges):
        dct = defaultdict(list)
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        dfn = [0]*n
        low = [0]*n
        index = 1
        stack = []
        part = []
        def tarjan(i, father):
            """tarjan算法"""
            nonlocal index, stack, part
            dfn[i] = low[i] = index
            index += 1
            stack.append(i)
            for j in dct[i]:
                if j==father:
                    continue
                if not dfn[j]:
                    tarjan(j, i)
                    low[i] = min(low[i], low[j])  # 割点 low[i] < dfn[i]
                                                # 割边 low[i] <= dfn[j]
                else:
                    low[i] = min(low[i], dfn[j])
            lst = []
            if dfn[i] == low[i]: # 连通分量
                while stack[-1] != i:
                    lst.append(stack.pop(-1))
                lst.append(stack.pop(-1))
            if len(lst) > 1:
                part = lst
            return
        tarjan(0, -1)

        # dijstra算法
        ans = [-1]*n
        stack = []
        for i in part:
            heapq.heappush(stack, [0, i])
        while stack:
            cur = heapq.heappop(stack)
            if ans[cur[1]] != -1:
                continue
            ans[cur[1]] = cur[0]
            for j in dct[cur[1]]:
                if ans[j] == -1:
                    heapq.heappush(stack, [cur[0]+1, j])
        return ans