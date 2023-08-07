from math import inf
import sys
import random
import sys
from collections import defaultdict
from math import inf
from typing import List, Set, Tuple, DefaultDict


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
        return random.randint(0, 10 ** 9 + 7)


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

    @staticmethod
    def get_point_doubly_connected_component_bfs(n: int, edge: List[List[int]]) -> Tuple[int, DefaultDict[int, Set[int]], List[Set[int]]]:
        # 模板：Tarjan求解无向图的点双连通分量

        dfs_id = 0
        order, low = [inf] * n, [inf] * n
        visit = [False] * n
        out = []
        parent = [-1]*n
        group_id = 0  # 点双个数
        group_node = defaultdict(set)  # 每个点双包含哪些点
        node_group_id = [set() for _ in range(n)]  # 每个点属于哪些点双，属于多个点双的点就是割点
        child = [0]*n
        for node in range(n):
            if not visit[node]:
                stack = [[node, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = True
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1

                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            # 遇到了割点（有根和非根两种）
                            if (parent == -1 and child[cur] > 1) or (parent != -1 and low[nex] >= order[cur]):
                                while out:
                                    top = out.pop()
                                    group_node[group_id].add(top[0])
                                    group_node[group_id].add(top[1])
                                    node_group_id[top[0]].add(group_id)
                                    node_group_id[top[1]].add(group_id)
                                    if top == (cur, nex):
                                        break
                                group_id += 1
                            # 我们将深搜时遇到的所有边加入到栈里面，当找到一个割点的时候
                            # 就将这个割点往下走到的所有边弹出，而这些边所连接的点就是一个点双
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            out.append((cur, nex))
                            child[cur] += 1
                            stack.append([nex, 0])
                        elif low[cur] > order[nex]:
                            low[cur] = order[nex]
                            out.append((cur, nex))
            if out:
                while out:
                    top = out.pop()
                    group_node[group_id].add(top[0])
                    group_node[group_id].add(top[1])
                    node_group_id[top[0]].add(group_id)
                    node_group_id[top[1]].add(group_id)
                group_id += 1
        # 点双的数量，点双分组节点，每个结点对应的点双编号（割点属于多个点双）
        return group_id, group_node, node_group_id

    def get_edge_doubly_connected_component_bfs(self, n: int, edge: List[Set[int]]):
        # 模板：Tarjan求解无向图的边双连通分量
        _, cutting_edges = self.get_cutting_point_and_cutting_edge_bfs(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)

        # 将所有的割边删掉剩下的都是边双连通分量，处理出割边，再对整个无向图进行一次BFS
        visit = [0]*n
        ans = []
        for i in range(n):
            if visit[i]:
                continue
            stack = [i]
            visit[i] = 1
            cur = [i]
            while stack:
                x = stack.pop()
                for j in edge[x]:
                    if not visit[j]:
                        visit[j] = 1
                        stack.append(j)
                        cur.append(j)
            ans.append(cur[:])
        # 边双的节点分组
        return ans

    @staticmethod
    def get_cutting_point_and_cutting_edge_bfs(n: int, edge: List[List[int]]) -> (Set[int], Set[Tuple[int, int]]):
        # 模板：Tarjan求解无向图的割点和割边（也就是桥）
        order, low = [inf] * n, [inf] * n
        visit = [0] * n
        cutting_point = set()
        cutting_edge = []
        child = [0]*n
        parent = [-1]*n
        dfs_id = 0
        for i in range(n):
            if not visit[i]:
                stack = [[i, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            pa = parent[cur]
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            if low[nex] > order[cur]:
                                cutting_edge.append((cur, nex) if cur < nex else (nex, cur))
                            if pa != -1 and low[nex] >= order[cur]:
                                cutting_point.add(cur)
                            elif pa == -1 and child[cur] > 1:  # 出发点没有祖先，所以特判一下
                                cutting_point.add(cur)
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            child[cur] += 1
                            stack.append([nex, 0])
                        else:
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]  # 注意这里是order
        # 割点和割边
        return cutting_point, cutting_edge


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        # 模板：强连通分量模板题
        for _ in range(ac.read_int()):
            n = ac.read_int()
            p = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for i in range(n):
                dct[i].append(p[i]-1)
            _, group, _ = TarjanCC().get_strongly_connected_component_bfs(n, dct)
            ans = [0]*n
            for g in group:
                x = len(group[g])
                for i in group[g]:
                    ans[i] = x
            ac.lst(ans)
        return


Solution().main()
