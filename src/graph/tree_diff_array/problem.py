"""

算法：LCA、倍增算法、树链剖分、树的质心、树的重心、离线LCA与树上差分
功能：来求一棵树的最近公共祖先（LCA）也可以使用
题目：

===================================力扣===================================
1483. 树节点的第 K 个祖先（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）动态规划与二进制跳转维护祖先信息，类似ST表的思想与树状数组的思想，经典LCA应用题
2646. 最小化旅行的价格总和（https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/）离线LCA与树上差分计数，再使用树形DP计算

===================================洛谷===================================
P3379 【模板】最近公共祖先（LCA）（https://www.luogu.com.cn/problem/P3379）最近公共祖先模板题
P7128 「RdOI R1」序列(sequence)（https://www.luogu.com.cn/problem/P7128）完全二叉树进行LCA路径模拟交换，使得数组有序
P3128 [USACO15DEC]Max Flow P（https://www.luogu.com.cn/problem/P3128）离线LCA与树上差分
P7167 [eJOI2020 Day1] Fountain（https://www.luogu.com.cn/problem/P7167）单调栈建树倍增在线LCA查询
P3384 【模板】重链剖分/树链剖分（https://www.luogu.com.cn/problem/P3384）树链剖分与树状数组模拟
P2912 [USACO08OCT]Pasture Walking G（https://www.luogu.com.cn/problem/P2912）离线LCA查询与任意点对之间距离计算
P3019 [USACO11MAR]Meeting Place S（https://www.luogu.com.cn/problem/P3019）离线查询 LCA 最近公共祖先
P3258 [JLOI2014]松鼠的新家（https://www.luogu.com.cn/problem/P3258）离线LCA加树上差分加树形DP
P6869 [COCI2019-2020#5] Putovanje（https://www.luogu.com.cn/problem/P6869）离线 LCA 查询与树上边差分计算

==================================LibreOJ==================================
#10135. 「一本通 4.4 练习 2」祖孙询问（https://loj.ac/p/10135）lca查询与判断

================================CodeForces================================
E. Tree Queries（https://codeforces.com/problemset/problem/1328/E）利用 LCA 判定节点组是否符合条件，也可以使用 dfs 序
C. Ciel the Commander（https://codeforces.com/problemset/problem/321/C）使用树的质心递归，依次切割形成平衡树赋值
E. A and B and Lecture Rooms（https://codeforces.com/problemset/problem/519/E）LCA经典运用题目，查询距离与第k个祖先节点，与子树节点计数
F. Berland Beauty（https://codeforces.com/contest/1296/problem/F）使用离线或者在线查询lca贪心构造，正解可能为倍增

================================AtCoder================================
D - Transit Tree Path（https://atcoder.jp/contests/abc070/tasks/abc070_d）典型LCA查询运用题，也可离线实现

================================AcWing================================
4202. 穿过圆（https://www.acwing.com/problem/content/4205/）使用位运算进行计算，也可使用包含关系建树，查询LCA计算距离


参考：
CSDN（https://blog.csdn.net/weixin_42001089/article/details/83590686）

"""
from typing import List

from src.graph.tree_diff_array.template import TreeDiffArray
from src.graph.tree_lca.template import OfflineLCA
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_6738(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:

        # 模板：离线LCA加树上差分加树形DP
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # 离线LCA
        res = OfflineLCA().bfs_iteration(dct, trips)
        # res = OfflineLCA().dfs_recursion(dct, trips)   # 也可以使用递归

        # 树上差分
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以使用递归

        # 迭代版的树形DP
        stack = [0]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
            else:
                i = ~i
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])

    @staticmethod
    def lg_p3128(ac=FastIO()):
        # 模板：离线LCA加树上差分
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        queries = [ac.read_list_ints_minus_one() for _ in range(k)]
        res = OfflineLCA().bfs_iteration(dct, queries)
        # res = OfflineLCA().dfs_recursion(dct, trips)  # 也可以使用递归
        queries = [queries[i] + [res[i]] for i in range(k)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以使用递归
        ac.st(max(cnt))
        return

    @staticmethod
    def lc_2646(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        # 模板：离线LCA与树上差分计数，再使用树形 DP 计算
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        res = OfflineLCA().bfs_iteration(dct, trips)
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)

        stack = [[0, 1]]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append([j, 1])
            else:
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])
