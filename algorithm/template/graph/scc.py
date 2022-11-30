"""
强连通分量
简介
在阅读下列内容之前，请务必了解 图论相关概念 中的基础部分。

强连通的定义是：有向图 G 强连通是指，G 中任意两个结点连通。

强连通分量（Strongly Connected Components，SCC）的定义是：极大的强连通子图。

这里要介绍的是如何来求强连通分量。



Tarjan 算法求强连通分量
在 Tarjan 算法中为每个结点  维护了以下几个变量：

：深度优先搜索遍历时结点  被搜索的次序。
：在  的子树中能够回溯到的最早的已经在栈中的结点。设以  为根的子树为 。 定义为以下结点的  的最小值： 中的结点；从  通过一条不在搜索树上的边能到达的结点。
一个结点的子树内结点的 dfn 都大于该结点的 dfn。

从根开始的一条路径上的 dfn 严格递增，low 严格非降。

按照深度优先搜索算法搜索的次序对图中所有的结点进行搜索，维护每个结点的 dfn 与 low 变量，且让搜索到的结点入栈。每当找到一个强连通元素，就按照该元素包含结点数目让栈中元素出栈。在搜索过程中，对于结点  和与其相邻的结点 （ 不是  的父节点）考虑 3 种情况：

 未被访问：继续对  进行深度搜索。在回溯过程中，用  更新 。因为存在从  到  的直接路径，所以  能够回溯到的已经在栈中的结点， 也一定能够回溯到。
 被访问过，已经在栈中：根据 low 值的定义，用  更新 。
 被访问过，已不在栈中：说明  已搜索完毕，其所在连通分量已被处理，所以不用对其做操作。


Kosaraju 算法
# # 引入
# # Kosaraju 算法最早在 1978 年由 S. Rao Kosaraju 在一篇未发表的论文上提出，但 Micha Sharir 最早发表了它。
# #
# # 过程
# # 该算法依靠两次简单的 DFS 实现：
# #
# # 第一次 DFS，选取任意顶点作为起点，遍历所有未访问过的顶点，并在回溯之前给顶点编号，也就是后序遍历。
# #
# # 第二次 DFS，对于反向后的图，以标号最大的顶点作为起点开始 DFS。这样遍历到的顶点集合就是一个强连通分量。对于所有未访问过的结点，选取标号最大的，重复上述过程。
# #
# # 两次 DFS 结束后，强连通分量就找出来了，Kosaraju 算法的时间复杂度为 。

Garbow 算法
过程
Garbow 算法是 Tarjan 算法的另一种实现，Tarjan 算法是用 dfn 和 low 来计算强连通分量的根，Garbow 维护一个节点栈，并用第二个栈来确定何时从第一个栈中弹出属于同一个强连通分量的节点。从节点  开始的 DFS 过程中，当一条路径显示这组节点都属于同一个强连通分量时，只要栈顶节点的访问时间大于根节点  的访问时间，就从第二个栈中弹出这个节点，那么最后只留下根节点 。在这个过程中每一个被弹出的节点都属于同一个强连通分量。

当回溯到某一个节点  时，如果这个节点在第二个栈的顶部，就说明这个节点是强连通分量的起始节点，在这个节点之后搜索到的那些节点都属于同一个强连通分量，于是从第一个栈中弹出那些节点，构成强连通分量。


应用
我们可以将一张图的每个强连通分量都缩成一个点。

然后这张图会变成一个 DAG，可以进行拓扑排序以及更多其他操作。

举个简单的例子，求一条路径，可以经过重复结点，要求经过的不同结点数量最多。
参考链接：https://oi-wiki.org/graph/scc/
参考题目：2360. 图中的最长环 https://leetcode.cn/problems/longest-cycle-in-a-graph/
参考题目：P3387	【模板】缩点 https://www.luogu.com.cn/problem/solution/P3387
"""


class Kosaraju:
    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.g2 = [[] for _ in range(self.n)]
        self.vis = [False] * n
        self.s = []
        self.color = [0] * n
        self.sccCnt = 0
        self.gen_reverse_graph()
        self.kosaraju()

    def gen_reverse_graph(self):
        for i in range(self.n):
            for j in self.g[i]:
                self.g2[j].append(i)
        return

    def dfs1(self, u):
        self.vis[u] = True
        for v in self.g[u]:
            if not self.vis[v]:
                self.dfs1(v)
        self.s.append(u)
        return

    def dfs2(self, u):
        self.color[u] = self.sccCnt
        for v in self.g2[u]:
            if not self.color[v]:
                self.dfs2(v)
        return

    def kosaraju(self):
        for i in range(self.n):
            if not self.vis[i]:
                self.dfs1(i)
        for i in range(self.n - 1, -1, -1):
            if not self.color[self.s[i]]:
                self.sccCnt += 1
                self.dfs2(self.s[i])
        self.color = [c-1 for c in self.color]
        return


class Tarjan:
    def __init__(self, edge):
        self.edge = edge
        self.n = len(edge)
        self.dfn = [0] * self.n
        self.low = [0] * self.n
        self.visit = [0] * self.n
        self.stamp = 0
        self.visit = [0]*self.n
        self.stack = []
        self.scc = []
        for i in range(self.n):
            if not self.visit[i]:
                self.tarjan(i)

    def tarjan(self, u):
        self.dfn[u], self.low[u] = self.stamp, self.stamp
        self.stamp += 1
        self.stack.append(u)
        self.visit[u] = 1
        for v in self.edge[u]:
            if not self.visit[v]:  # 未访问
                self.tarjan(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.visit[v] == 1:
                self.low[u] = min(self.low[u], self.dfn[v])

        if self.dfn[u] == self.low[u]:
            cur = []
            # 栈中u之后的元素是一个完整的强连通分量
            while True:
                cur.append(self.stack.pop())
                self.visit[cur[-1]] = 2  # 节点已弹出，归属于现有强连通分量
                if cur[-1] == u:
                    break
            self.scc.append(cur)
        return
        
        
def test_directed_graph():
    # 有向无环图
    edge = [[1, 2], [], [3], []]
    n = 4
    kosaraju = Kosaraju(n, edge)
    assert len(set(kosaraju.color)) == 4
    tarjan = Tarjan(edge)
    assert len(tarjan.scc) == 4

    # 有向有环图
    edge = [[1, 2], [2], [0, 3], []]
    n = 4
    kosaraju = Kosaraju(n, edge)
    assert len(set(kosaraju.color)) == 2
    tarjan = Tarjan(edge)
    assert len(tarjan.scc) == 2
    return


if __name__ == '__main__':
    test_directed_graph()
