"""

Algorithm：LCA、倍增算法、树链剖分、树的质心、树的重心、离线LCA与树上差分
Function：来求一棵树的最近公共祖先（LCA）也可以

====================================LeetCode====================================
1483（https://leetcode.com/problems/kth-ancestor-of-a-tree-node/）动态规划与二进制跳转维护祖先信息，类似ST表的思想与树状数组的思想，LCA应用题
2646（https://leetcode.com/problems/minimize-the-total-price-of-the-trips/）离线LCA与树上差分counter，再树形DP

=====================================LuoGu======================================
3379（https://www.luogu.com.cn/problem/P3379）最近公共祖先模板题
7128（https://www.luogu.com.cn/problem/P7128）完全二叉树LCA路径implemention交换，使得数组有序
3128（https://www.luogu.com.cn/problem/P3128）离线LCA与树上差分
7167（https://www.luogu.com.cn/problem/P7167）monotonic_stack|建树倍增在线LCA查询
3384（https://www.luogu.com.cn/problem/P3384）树链剖分与树状数组implemention
2912（https://www.luogu.com.cn/problem/P2912）离线LCA查询与任意点对之间距离
3019（https://www.luogu.com.cn/problem/P3019）offline_query LCA 最近公共祖先
3258（https://www.luogu.com.cn/problem/P3258）离线LCA|树上差分|树形DP
6869（https://www.luogu.com.cn/problem/P6869）offline_lca 查询与树上边差分

==================================LibreOJ==================================
#10135. 「一本通 4.4 练习 2」祖孙询问（https://loj.ac/p/10135）lca查询与判断

===================================CodeForces===================================
1328E（https://codeforces.com/problemset/problem/1328/E）利用 LCA 判定节点组是否符合条件，也可以 dfs 序
321C（https://codeforces.com/problemset/problem/321/C）树的质心递归，依次切割形成平衡树赋值
519E（https://codeforces.com/problemset/problem/519/E）LCA运用题目，查询距离与第k个祖先节点，与子树节点counter
1296F（https://codeforces.com/contest/1296/problem/F）离线或者在线查询lcagreedyconstruction，正解可能为倍增

====================================AtCoder=====================================
D - Transit Tree Path（https://atcoder.jp/contests/abc070/tasks/abc070_d）典型LCA查询运用题，也可离线实现

=====================================AcWing=====================================
4202（https://www.acwing.com/problem/content/4205/）bit_operation，也可包含关系建树，查询LCA距离


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

        # 离线LCA|树上差分|树形DP
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # 离线LCA
        res = OfflineLCA().bfs_iteration(dct, trips)
        # res = OfflineLCA().dfs_recursion(dct, trips)   # 也可以递归

        # 树上差分
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以递归

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
        # 离线LCA|树上差分
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        queries = [ac.read_list_ints_minus_one() for _ in range(k)]
        res = OfflineLCA().bfs_iteration(dct, queries)
        # res = OfflineLCA().dfs_recursion(dct, trips)  # 也可以递归
        queries = [queries[i] + [res[i]] for i in range(k)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以递归
        ac.st(max(cnt))
        return

    @staticmethod
    def lc_2646(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        # 离线LCA与树上差分counter，再树形 DP 
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