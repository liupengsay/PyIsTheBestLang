"""
Algorithm：bfs|deque_bfs|01-bfs|discretization_bfs|bound_bfs|coloring_method|odd_circle
Description：multi_source_bfs|bilateral_bfs|spfa|a-star|heuristic_search

====================================LeetCode====================================


=====================================LuoGu======================================
P1099（https://www.luogu.com.cn/problem/P1099）tree_diameter|bfs|two_pointers|monotonic_queue|classical
P2491（https://www.luogu.com.cn/problem/P2491）tree_diameter|bfs|two_pointers|monotonic_queue|classical


===================================CodeForces===================================

====================================AtCoder=====================================

=====================================AcWing=====================================

"""

from typing import List

from src.graph.tree_diameter.template import TreeDiameter
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1805d(ac=FastIO()):
        # tree_diameter与端点距离，节点对距离至少为k的连通块个数
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append([v, 1])
            edge[v].append([u, 1])
        tree = TreeDiameter(edge)
        u, v = tree.get_diameter_info()[:2]
        dis1, _ = tree.get_bfs_dis(u)
        dis2, _ = tree.get_bfs_dis(v)
        diff = [0] * n
        for i in range(n):
            diff[ac.max(dis1[i], dis2[i])] += 1
        diff[0] = 1
        diff = ac.accumulate(diff)[1:]
        ac.lst([ac.min(x, n) for x in diff])
        return

    @staticmethod
    def lg_p3304(ac=FastIO()):
        # 带权无向图的tree_diameter以及tree_diameter的必经边
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        original = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append([j, k])
            dct[j].append([i, k])
            original[i][j] = original[j][i] = k
        # 首先tree_diameter
        tree = TreeDiameter(dct)
        x, y, path, dia = tree.get_diameter_info()
        ac.st(dia)
        # 确定tree_diameter上每个点的最远端距离
        nodes = set(path)
        dis = [0] * n
        for x in path:
            q = [[x, -1, 0]]
            while q:
                i, fa, d = q.pop()
                for j, w in dct[i]:
                    if j != fa and j not in nodes:
                        dis[x] = d + w
                        q.append([j, i, d + w])

        # tree_diameter必经边的最右边端点
        m = len(path)
        pre = right = 0
        for j in range(1, m):
            pre += original[path[j - 1]][path[j]]
            right = j
            if dis[path[j]] == dia - pre:  # 此时点下面有非当前tree_diameter的最远路径
                break

        # tree_diameter必经边的最左边端点
        left = m - 1
        post = 0
        for j in range(m - 2, -1, -1):
            post += original[path[j]][path[j + 1]]
            left = j
            if dis[path[j]] == dia - post:  # 此时点下面有非当前tree_diameter的最远路径
                break

        ans = ac.max(0, right - left)
        ac.st(ans)
        return

    @staticmethod
    def lc_1617(n: int, edges: List[List[int]]) -> List[int]:
        # brute_force子集union_find判断连通性再tree_diameter
        ans = [0] * n
        for state in range(1, 1 << n):
            node = [i for i in range(n) if state & (1 << i)]
            ind = {num: i for i, num in enumerate(node)}
            m = len(node)
            dct = [[] for _ in range(m)]
            uf = UnionFind(m)
            for u, v in edges:
                u -= 1
                v -= 1
                if u in ind and v in ind:
                    dct[ind[u]].append([ind[v], 1])
                    dct[ind[v]].append([ind[u], 1])
                    uf.union(ind[u], ind[v])
            if uf.part != 1:
                continue
            tree = TreeDiameter(dct)
            ans[tree.get_diameter_info()[-1]] += 1
        return ans[1:]

    @staticmethod
    def lq_5890(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        ans = 0
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w])
            dct[v].append([u, w])
            ans += w * 2
        dis = TreeDiameter(dct).get_diameter_info()[-1]
        ans -= dis
        ac.st(ans)
        return
