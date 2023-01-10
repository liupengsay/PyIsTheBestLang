from typing import List
import unittest

"""
# Tarjan

## 算法功能
Tarjan 算法是基于深度优先搜索的算法，用于求解图的连通性问题，参考[60 分钟搞定图论中的 Tarjan 算法]

- Tarjan 算法可以在线性时间内求出**无向图的割点与桥**，进一步地可以求解**无向图的双连通分量**
- Tarjan 算法可以也可以求解**有向图的强连通分量**，进一步地可以**求有向图的必经点与必经边**

## 可以求有向图与无向图的割点、割边、点双连通分量与边双连通分量
[60 分钟搞定图论中的 Tarjan 算法]: https://zhuanlan.zhihu.com/p/101923309

## 算法伪代码

## 算法模板与测试用例
- 见Tarjan.py

## 经典题目
- 无向有环图求割点[1568. 使陆地分离的最少天数]
- 无向有环图求点最近的环[2204. Distance to a Cycle in Undirected Graph]
- 无向有环图求割边[1192. 查找集群内的「关键连接」]
- 有向有环图求环[2360. 图中的最长环]

[1192. 查找集群内的「关键连接」]: https://leetcode.cn/problems/critical-connections-in-a-network/solution/by-liupengsay-dlc2/
[2360. 图中的最长环]: https://leetcode.cn/problems/longest-cycle-in-a-graph/solution/by-liupengsay-4ff6/
[2204. Distance to a Cycle in Undirected Graph]: https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/solution/er-xu-cheng-ming-jiu-xu-zui-python3tarja-09qn/
[1568. 使陆地分离的最少天数]: https://leetcode.cn/problems/minimum-number-of-days-to-disconnect-island/solution/by-liupengsay-zd7w/
P8436 【模板】边双连通分量：https://www.luogu.com.cn/problem/P8436（有自环与重边，通过虚拟节点进行扩边）
P8435 【模板】点双连通分量：https://www.luogu.com.cn/problem/P8435（有自环与重边，通过虚拟节点进行扩边）
P1656 炸铁路（https://www.luogu.com.cn/problem/P1656）求割边
P1793 跑步（https://www.luogu.com.cn/problem/P1793）求连通图两个指定点之间的割点，使用枚举与并查集的方式进行求解

"""


class Tarjan:
    def __init__(self):
        return

    @staticmethod
    def check_graph(edge, n):
        # edge: 边连接关系 [[],..] n:节点数

        # 访问序号与根节点序号
        visit = [0] * n
        root = [0] * n
        # 割点
        cut_node = []
        # 割边
        cut_edge = []
        # 强连通分量子树
        sub_group = []

        # 中间变量
        stack = []
        index = 1
        in_stack = [0] * n

        def tarjan(i, father):
            nonlocal index
            visit[i] = root[i] = index
            index += 1
            stack.append(i)

            in_stack[i] = 1
            child = 0
            for j in edge[i]:
                if j != father:
                    if not visit[j]:
                        child += 1
                        tarjan(j, i)
                        x, y = root[i], root[j]
                        root[i] = x if x < y else y
                        # 割边 low[i] < dfn[i]
                        if visit[i] < root[j]:
                            cut_edge.append(sorted([i, j]))
                        # 两种情况下才为割点 low[i] <= dfn[i]
                        if father != -1 and visit[i] <= root[j]:
                            cut_node.append(i)
                        elif father == -1 and child >= 2:
                            cut_node.append(i)
                    elif in_stack[j]:
                        x, y = root[i], visit[j]
                        root[i] = x if x < y else y

            if root[i] == visit[i]:
                lst = []
                while stack[-1] != i:
                    lst.append(stack.pop())
                    in_stack[lst[-1]] = 0
                lst.append(stack.pop())
                in_stack[lst[-1]] = 0
                r = min(root[ls] for ls in lst)
                for ls in lst:
                    root[ls] = r
                lst.sort()
                sub_group.append(lst[:])
            return

        for k in range(n):
            if not visit[k]:
                tarjan(k, -1)
        cut_edge.sort()
        cut_node.sort()
        sub_group.sort()
        return cut_edge, cut_node, sub_group


class TestGeneral(unittest.TestCase):
    def test_undirected_graph(self):
        # 无向无环图
        edge = [[1, 2], [0, 3], [0, 3], [1, 2]]
        n = 4
        ta = Tarjan()
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert not cut_edge
        assert not cut_node
        assert sub_group == [[0, 1, 2, 3]]

        # 无向有环图
        edge = [[1, 2, 3], [0, 2], [0, 1], [0]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 3]]
        assert cut_node == [0]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有自环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2, 3]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]
        return

    def test_directed_graph(self):
        # 有向无环图
        edge = [[1, 2], [], [3], []]
        n = 4
        ta = Tarjan()
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 1], [0, 2], [2, 3]]
        assert cut_node == [0, 2]
        assert sub_group == [[0], [1], [2], [3]]

        edge = [[1, 2], [2], [3], []]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 1], [1, 2], [2, 3]]
        assert cut_node == [1, 2]
        assert sub_group == [[0], [1], [2], [3]]

        # 有向有环图
        edge = [[1, 2], [2], [0, 3], []]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]
        return


if __name__ == '__main__':
    unittest.main()
