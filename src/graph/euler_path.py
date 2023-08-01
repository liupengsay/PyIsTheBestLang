import unittest
from typing import List

from src.fast_io import FastIO
from src.graph.union_find import UnionFind

"""

算法：欧拉路径（使用深度优先搜索里面的Hierholzer算法）
功能：求解有向图与无向图中的欧拉路径，定义比较复杂且不统一，须根据实际情况作适配与调整
有向图欧拉路径：图中恰好存在 1 个点出度比入度多 1（这个点即为起点） 1 个点出度比入度少 1（这个点即为终点）其余相等
有向图欧拉回路：所有节点出度等于入度，起终点可以为任意点
无向图欧拉路径：图中恰好存在 2 个点的度数是奇数，其余节点的度数为偶数，这两个度数为奇数的点即为欧拉路径的起点和终点 
无向图欧拉回路：所有点的度数都是偶数（起点和终点可以为任意点）
哈密顿路径：类似欧拉路径，只是要求经过每个顶点恰好一次，使用回溯？
哈密顿回路：类似拉回路，只是要求经过每个顶点恰好一次，使用回溯？

注1：存在欧拉回路（即满足存在欧拉回路的条件），也一定存在欧拉路径，
注2：图有欧拉路径必须满足将它的有向边视为无向边后它是连通的（不考虑度为 0 的孤立点）连通性的判断我们可以使用并查集或 dfs 

题目：
===================================力扣===================================
332. 重新安排行程（https://leetcode.cn/problems/reconstruct-itinerary/）欧拉回路模板题
753. 破解保险箱（https://leetcode.cn/problems/cracking-the-safe/solution/er-xu-cheng-ming-jiu-xu-zui-by-liupengsa-lm77/）
2097. 合法重新排列数对（https://leetcode.cn/problems/valid-arrangement-of-pairs/submissions/）欧拉路径模板题，注意确定首尾点

===================================洛谷===================================
P7771 【模板】欧拉路径（https://www.luogu.com.cn/problem/P7771）欧拉路径模板题
P6066 [USACO05JAN]Watchcow S（https://www.luogu.com.cn/problem/P6066）欧拉路径模板题
P1127 词链（https://www.luogu.com.cn/problem/P1127）经过每个顶点一次有向边不确定且字典序最小（转换为有向图欧拉路径或者回路）
P2731 [USACO3.3]骑马修栅栏 Riding the Fences（https://www.luogu.com.cn/problem/P2731）经过每条确定无向边一次且字典序最小（需要使用邻接矩阵转换为无向图欧拉路径或者回路）
P1341 无序字母对（https://www.luogu.com.cn/problem/P1341）经过每条确定无向边一次且字典序最小（需要使用邻接矩阵转换为无向图欧拉路径或者回路）


参考：
OI WiKi（https://oi-wiki.org/graph/euler/）
https://www.jianshu.com/p/8394b8e5b878
https://www.luogu.com.cn/problem/solution/P7771
"""


class DirectedEulerPath:
    def __init__(self, n, pairs):
        # 数组形式存储的有向连接关系
        self.n = n
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = [0]*self.n
        # 存储图关系
        edge = [[] for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] -= 1
            edge[i].append(j)

        # 根据字典序优先访问较小的
        for i in range(self.n):
            edge[i].sort(reverse=True)

        # 寻找起始节点
        starts = []
        ends = []
        zero = 0
        for i in range(self.n):
            if degree[i] == 1:
                starts.append(i)
            elif degree[i] == -1:
                ends.append(i)
            else:
                zero += 1
        del degree

        # 图中恰好存在 1 个点出度比入度多 1（这个点即为起点） 1 个点出度比入度少 1（这个点即为终点）其余相等
        if not len(starts) == len(ends) == 1:
            if zero != self.n:
                return
            starts = [0]

        @FastIO.bootstrap
        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            while edge[pre]:
                nex = edge[pre].pop()
                yield dfs(nex)
                self.nodes.append(nex)
                self.paths.append([pre, nex])
            yield

        dfs(starts[0])
        # 注意判断所有边都经过的才算欧拉路径
        self.paths.reverse()
        self.nodes.append(starts[0])
        self.nodes.reverse()
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class UnDirectedEulerPath:
    def __init__(self, n, pairs):
        # 数组形式存储的有向连接关系
        self.n = n
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = [0]*self.n
        # 存储图关系
        edge = [[0]*self.n for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] += 1
            edge[i][j] += 1
            edge[j][i] += 1

        # 寻找起始节点
        starts = []
        zero = 0
        for i in range(self.n):
            # 如果有节点出度比入度恰好多 1，那么只有它才能是起始
            if degree[i] % 2:
                starts.append(i)
            else:
                zero += 1
        del degree

        # 不存在欧拉路径
        if not len(starts) == 2:
            if zero != self.n:
                return
            starts = [0]

        @FastIO.bootstrap
        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            for nex in range(self.n):
                while edge[pre][nex]:
                    edge[pre][nex] -= 1
                    edge[nex][pre] -= 1
                    yield dfs(nex)
                    self.nodes.append(nex)
                    self.paths.append([pre, nex])
            yield

        dfs(starts[0])
        # 注意判断所有边都经过的才算欧拉路径
        self.paths.reverse()
        self.nodes.append(starts[0])
        self.nodes.reverse()
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p7771(ac=FastIO()):
        # 模板：有向图欧拉路径或者欧拉回路
        n, m = ac.read_ints()
        # 存储图关系
        pairs = [ac.read_list_ints_minus_one() for _ in range(m)]
        # 注意可能包括自环与重边
        euler = DirectedEulerPath(n, pairs)
        if euler.exist:
            ac.lst([x + 1 for x in euler.nodes])
        else:
            ac.st("No")
        return

    @staticmethod
    def lg_p2731(ac=FastIO()):
        # 模板：无向图欧拉路径或者欧拉回路
        m = ac.read_int()
        pairs = [ac.read_list_ints() for _ in range(m)]
        node = set()
        for a, b in pairs:
            node.add(a)
            node.add(b)
        node = sorted(list(node))
        n = len(node)
        ind = {node[i]: i for i in range(n)}
        pairs = [[ind[x], ind[y]] for x, y in pairs]
        euler = UnDirectedEulerPath(n, pairs)
        for a in euler.nodes:
            ac.st(node[a])
        return

    @staticmethod
    def lg_p1341(ac=FastIO()):
        # 模板：无向图欧拉路径或者欧拉回路
        m = ac.read_int()
        nodes = set()
        pairs = []
        for _ in range(m):
            s = ac.read_str()
            nodes.add(s[0])
            nodes.add(s[1])
            pairs.append([s[0], s[1]])

        # 首先离散化编码判断是否连通
        nodes = sorted(list(nodes))
        ind = {num: i for i, num in enumerate(nodes)}
        n = len(nodes)
        uf = UnionFind(n)
        for x, y in pairs:
            uf.union(ind[x], ind[y])
        if uf.part != 1:
            ac.st("No Solution")
            return

        # 经典无向图计算字典序最小的欧拉序
        pairs = [[ind[x], ind[y]] for x, y in pairs]
        euler = UnDirectedEulerPath(n, pairs)
        if not euler.exist:
            ac.st("No Solution")
            return
        ans = "".join([nodes[a] for a in euler.nodes])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1127(ac=FastIO()):
        # 模板：有向图欧拉路径或者欧拉回路
        m = ac.read_int()

        # 最关键的建图
        nodes = set()
        pairs = []
        for _ in range(m):
            s = ac.read_str()
            nodes.add(s[0])
            nodes.add(s[-1])
            nodes.add(s)
            pairs.append([s[0], s])
            pairs.append([s, s[-1]])

        # 按照序号编码并检查连通性
        nodes = sorted(list(nodes))
        ind = {num: i for i, num in enumerate(nodes)}
        n = len(nodes)
        uf = UnionFind(n)
        for x, y in pairs:
            uf.union(ind[x], ind[y])
        if uf.part != 1:
            ac.st("***")
            return

        # 有向图欧拉路径或者欧拉回路的获取
        pairs = [[ind[x], ind[y]] for x, y in pairs]
        euler = DirectedEulerPath(n, pairs)
        if not euler.exist:
            ac.st("***")
            return

        # 去除虚拟开头与结尾字母
        ans = []
        for x in euler.nodes:
            ans.append(nodes[x])
        ans = ans[1:-1:2]
        ac.st(".".join(ans))
        return

    @staticmethod
    def lg_p6606(ac=FastIO()):
        # 模板：有向图欧拉路径或者欧拉回路
        n, m = ac.read_ints()
        # 最关键的建图
        pairs = []
        for _ in range(m):
            u, v = ac.read_ints_minus_one()
            # 每条边两个方向各走一遍
            pairs.append([u, v])
            pairs.append([v, u])

        # 有向图欧拉路径或者欧拉回路的获取
        euler = DirectedEulerPath(n, pairs)
        i = euler.nodes.index(0)
        for x in euler.nodes[i:] + euler.nodes[:i]:
            ac.st(x + 1)
        return

    @staticmethod
    def lc_2097(pairs: List[List[int]]) -> List[List[int]]:
        # 模板：欧拉路径模板题，离散化后转化为图的欧拉路径求解
        nodes = set()
        for a, b in pairs:
            nodes.add(a)
            nodes.add(b)
        nodes = list(nodes)
        n = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        lst = [[ind[a], ind[b]] for a, b in pairs]
        ep = DirectedEulerPath(n, lst)
        ans = ep.paths
        return [[nodes[x], nodes[y]] for x, y in ans]


class TestGeneral(unittest.TestCase):

    def test_euler_path(self):
        pairs = [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]
        ep = DirectedEulerPath(4, pairs)
        assert ep.paths == [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]

        pairs = [[1, 3], [2, 1], [4, 2], [3, 3], [1, 2], [3, 4]]
        ep = DirectedEulerPath(4, pairs)
        assert ep.nodes == [1, 2, 1, 3, 3, 4, 2]
        return


if __name__ == '__main__':
    unittest.main()
