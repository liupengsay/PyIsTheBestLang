from math import inf
import sys
import random
import sys
from collections import defaultdict
from math import inf
from typing import List, Set, DefaultDict


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_ints():
        return map(int, sys.stdin.readline().strip().split())

    @staticmethod
    def read_floats():
        return map(float, sys.stdin.readline().strip().split())

    @staticmethod
    def read_ints_minus_one():
        return map(lambda x: int(x) - 1, sys.stdin.readline().strip().split())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

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

    def ask(self, lst):
        # CF交互题输出询问并读取结果
        self.lst(lst)
        sys.stdout.flush()
        res = self.read_int()
        # 记得任何一个输出之后都要 sys.stdout.flush() 刷新
        return res

    def out_put(self, lst):
        # CF交互题输出最终答案
        self.lst(lst)
        sys.stdout.flush()
        return

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    @staticmethod
    def get_random_seed():
        # 随机种子避免哈希冲突
        return random.randint(0, 10**9+7)


class TarjanCC:
    def __init__(self):
        return

    @staticmethod
    def get_strongly_connected_component_bfs(n: int, edge: List[List[int]]) -> (int, DefaultDict[int, Set[int]], List[int]):
        # 模板：Tarjan求解有向图的强连通分量 edge为有向边要求无自环与重边
        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        out = []
        in_stack = [0] * n
        scc_id = 0
        scc_node_id = defaultdict(set)
        node_scc_id = [-1] * n
        parent = [-1]*n
        for node in range(n):
            if not visit[node]:
                stack = [[node, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                        out.append(cur)
                        in_stack[cur] = 1
                    if ind == len(edge[cur]):
                        stack.pop()
                        if order[cur] == low[cur]:
                            while out:
                                top = out.pop()
                                in_stack[top] = 0
                                scc_node_id[scc_id].add(top)
                                node_scc_id[top] = scc_id
                                if top == cur:
                                    break
                            scc_id += 1

                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if not visit[nex]:
                            parent[nex] = cur
                            stack.append([nex, 0])
                        elif in_stack[nex]:
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]  # 注意这里是order

        # 建立新图
        new_dct = [set() for _ in range(scc_id)]
        for i in range(n):
            for j in edge[i]:
                a, b = node_scc_id[i], node_scc_id[j]
                if a != b:
                    new_dct[b].add(a)
        new_degree = [0]*scc_id
        for i in range(scc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        # SCC的数量，分组，每个结点对应的SCC编号
        return scc_id, scc_node_id, node_scc_id


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_3813(ac=FastIO()):
        # 模板：强连通分量模板与拓扑排序DP
        n, m = ac.read_ints()
        s = ac.read_str()
        dct = [set() for _ in range(n)]
        for _ in range(m):
            a, b = ac.read_ints_minus_one()
            if a == b:
                ac.st(-1)
                return
            dct[a].add(b)
        scc_id, _, _ = TarjanCC().get_strongly_connected_component_bfs(n, [list(e) for e in dct])
        if scc_id != n:
            ac.st(-1)
            return

        cur = [[0]*26 for _ in range(n)]
        degree = [0]*n
        for i in range(n):
            for j in dct[i]:
                degree[j] += 1
        stack = [i for i in range(n) if not degree[i]]
        ans = 0
        while stack:
            nex = []
            for i in stack:
                cur[i][ord(s[i]) - ord("a")] += 1
                x = max(cur[i])
                if x > ans:
                    ans = x
                for j in dct[i]:
                    for w in range(26):
                        y = cur[i][w]
                        if y > cur[j][w]:
                            cur[j][w] = y
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex[:]
        ac.st(ans)
        return


Solution().main()
