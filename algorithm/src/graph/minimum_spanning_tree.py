import unittest
from typing import List

from algorithm.src.graph.union_find import UnionFind

"""

算法：最小生成树（Kruskal算法和Prim算法两种）
功能：计算无向图边权值和最小的生成树
Prim在稠密图中比Kruskal优，在稀疏图中比Kruskal劣。Prim是以更新过的节点的连边找最小值，Kruskal是直接将边排序。
两者其实都是运用贪心的思路，Kruskal相对比较常用

题目：

===================================力扣===================================
1489. 找到最小生成树里的关键边和伪关键边（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）计算最小生成树的关键边与伪关键边


===================================洛谷===================================
P3366 最小生成树（https://www.luogu.com.cn/problem/P3366）计算最小生成树的权值和

P2872 Building Roads S（https://www.luogu.com.cn/problem/P2872）使用prim计算最小生成树
P1991 无线通讯网（https://www.luogu.com.cn/problem/P1991）计算保证k个连通块下最小的边权值
P1661 扩散（https://www.luogu.com.cn/problem/P1661）最小生成树的边最大权值
P1547 [USACO05MAR]Out of Hay S（https://www.luogu.com.cn/problem/P1547）最小生成树的边最大权值
P2121 拆地毯（https://www.luogu.com.cn/problem/P2121）保留 k 条边的最大生成树权值
P2126 Mzc家中的男家丁（https://www.luogu.com.cn/problem/P2126）转化为最小生成树求解

P2330 [SCOI2005]繁忙的都市（https://www.luogu.com.cn/problem/P2330）最小生成树边数量与最大边权值
P2504 [HAOI2006]聪明的猴子（https://www.luogu.com.cn/problem/P2504）识别为最小生成树求解
P2700 逐个击破（https://www.luogu.com.cn/problem/P2700）逆向思维与最小生成树，选取最大权组合，修改并查集size

P1195 口袋的天空（https://www.luogu.com.cn/record/list?user=739032&status=12&page=13）最小生成树生成K个连通块
P1194 买礼物（https://www.luogu.com.cn/problem/P1194）最小生成树变种问题
P2820 局域网（https://www.luogu.com.cn/problem/P2820）最小生成树裸题

P2916 [USACO08NOV]Cheering up the Cow G（https://www.luogu.com.cn/problem/P2916）需要自定义排序之后计算最小生成树的好题
P4955 [USACO14JAN]Cross Country Skiing S（https://www.luogu.com.cn/problem/P4955）最小生成树，自定义中止条件
P6705 [COCI2010-2011#7] POŠTAR（https://www.luogu.com.cn/problem/P6705）枚举最小值，使用最小生成树，与自定义权值进行计算
P7775 [COCI2009-2010#2] VUK（https://www.luogu.com.cn/problem/P7775）BFS加最小生成树思想，求解

P2658 汽车拉力比赛（https://www.luogu.com.cn/problem/P2658）典型最小生成树计算

参考：OI WiKi（xx）
"""


class MininumSpanningTree:
    def __init__(self, edges, n):
        # n个节点
        self.n = n
        # m条权值边edges
        self.edges = edges
        self.cost = 0
        self.gen_minimum_spanning_tree()
        return

    def gen_minimum_spanning_tree(self):
        self.edges.sort(key=lambda item: item[2])
        # 贪心按照权值选择边进行连通合并
        uf = UnionFind(self.n)
        for x, y, z in self.edges:
            if uf.union(x, y):
                self.cost += z
        # 不能形成生成树
        if uf.part != 1:
            self.cost = -1
        return


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1489(n: int, edges: List[List[int]]) -> List[List[int]]:
        # 模板：求最小生成树的关键边与伪关键边
        m = len(edges)
        # 代价排序
        lst = list(range(m))
        lst.sort(key=lambda it: edges[it][2])

        # 计算最小生成树代价
        min_cost = 0
        uf = UnionFind(n)
        for i in lst:
            x, y, cost = edges[i]
            if uf.union(x, y):
                min_cost += cost

        # 枚举关键边
        key = set()
        for i in lst:
            cur_cost = 0
            uf = UnionFind(n)
            for j in lst:
                if j != i:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
            if cur_cost > min_cost or uf.part != 1:
                key.add(i)

        # 枚举伪关键边
        fake = set()
        for i in lst:
            if i not in key:
                cur_cost = edges[i][2]
                uf = UnionFind(n)
                # 先将当前边加入生成树
                uf.union(edges[i][0], edges[i][1])
                for j in lst:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
                # 若仍然是最小生成树就说明是伪关键边
                if cur_cost == min_cost and uf.part == 1:
                    fake.add(i)

        return [list(key), list(fake)]


    @staticmethod
    def lg_p2872():
        # https://www.luogu.com.cn/record/74793627

        def main():
            n, m = map(int, input().split())
            edge = [[0] * (n + 1) for _ in range(n + 1)]
            vtx = [[]]
            for _ in range(n):
                vtx.append([int(i) for i in input().split()])
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    edge[i][j] = edge[j][i] = ((vtx[i][0] - vtx[j][0]) ** 2 +
                                               (vtx[i][1] - vtx[j][1]) ** 2) ** 0.5
            for _ in range(m):
                a, b = map(int, input().split())
                edge[a][b] = edge[b][a] = 0

            # prim计算最小生成树

            def Prim():
                vis = set([1])
                dist = edge[1].copy()
                values = 0
                for k in range(n - 1):
                    next_v = -1
                    min_d = float('inf')
                    for i in range(1, n + 1):
                        if dist[i] < min_d and i not in vis:
                            next_v = i
                            min_d = dist[i]
                    vis.add(next_v)
                    values += min_d
                    for j in range(1, n + 1):
                        if dist[j] > edge[next_v][j] and j not in vis:
                            dist[j] = edge[next_v][j]
                return values

            print('{:.2f}'.format(Prim()))

        main()
        return


class TestGeneral(unittest.TestCase):

    def test_minimum_spanning_tree(self):
        n = 3
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MininumSpanningTree(edges, n)
        assert mst.cost == 5

        n = 4
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MininumSpanningTree(edges, n)
        assert mst.cost == -1
        return


if __name__ == '__main__':
    unittest.main()
