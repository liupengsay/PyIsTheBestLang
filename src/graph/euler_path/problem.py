"""

Algorithm：euler_path|hierholzer
Description：directed_graph|undirected_graph|euler_path
directed_euler_path：start point has out_degree - in_degree = 1, end point has in_degree - out_degree = 1, others in_degree = out_degree
directed_euler_circular_path：all points have in_degree = out_degree, all points can be start or end point
undirected_euler_path：start and end point have odd degree, others have even degree
undirected_euler_circular_path：all points have even degree, all points can be start or end point
hamilton_path：like euler_path, path pass every node exactly once, back_track?
hamilton_circular_path：like euler_circular_path, path pass every node exactly once, back_track?
Note1：where there exist euler_circular_path, there exist euler_path
Note2：where there exist euler_path if and only if the graph is connected

====================================LeetCode====================================
332（https://leetcode.cn/problems/reconstruct-itinerary/）euler_circular_path
753（https://leetcode.cn/problems/cracking-the-safe/）euler_path
2097（https://leetcode.cn/problems/valid-arrangement-of-pairs/submissions/）euler_path
1743（https://leetcode.cn/problems/restore-the-array-from-adjacent-pairs/）undirected_euler_path|discretization

=====================================LuoGu======================================
P7771（https://www.luogu.com.cn/problem/P7771）euler_path
P6066（https://www.luogu.com.cn/problem/P6066）euler_path
P1127（https://www.luogu.com.cn/problem/P1127）lexicographical_order_minimum|directed_euler_path|specific_plan
P2731（https://www.luogu.com.cn/problem/P2731）lexicographical_order_minimum|undirected_euler_path|specific_plan
P1341（https://www.luogu.com.cn/problem/P1341）lexicographical_order_minimum|undirected_euler_path|specific_plan

=====================================AcWing=====================================
4211（https://www.acwing.com/problem/content/4214/）directed_euler_path|specific_plan

OI WiKi（https://oi-wiki.org/graph/euler/）
https://www.jianshu.com/p/8394b8e5b878
https://www.luogu.com.cn/problem/solution/P7771
"""
from typing import List

from src.graph.euler_path.template import DirectedEulerPath, UnDirectedEulerPath
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p7771(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7771
        tag: euler_path
        """
        # 有向图euler_path或者euler_circular_path
        n, m = ac.read_list_ints()
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
        """
        url: https://www.luogu.com.cn/problem/P2731
        tag: lexicographical_order_minimum|undirected_euler_path|specific_plan
        """
        # 无向图euler_path或者euler_circular_path
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
        """
        url: https://www.luogu.com.cn/problem/P1341
        tag: lexicographical_order_minimum|undirected_euler_path|specific_plan
        """
        # 无向图euler_path或者euler_circular_path
        m = ac.read_int()
        nodes = set()
        pairs = []
        for _ in range(m):
            s = ac.read_str()
            nodes.add(s[0])
            nodes.add(s[1])
            pairs.append([s[0], s[1]])

        # 首先discretization编码判断是否连通
        nodes = sorted(list(nodes))
        ind = {num: i for i, num in enumerate(nodes)}
        n = len(nodes)
        uf = UnionFind(n)
        for x, y in pairs:
            uf.union(ind[x], ind[y])
        if uf.part != 1:
            ac.st("No Solution")
            return

        # 无向图lexicographical_order最小的欧拉序
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
        """
        url: https://www.luogu.com.cn/problem/P1127
        tag: lexicographical_order_minimum|directed_euler_path|specific_plan
        """
        # 有向图euler_path或者euler_circular_path
        m = ac.read_int()

        # 最关键的build_graph|
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

        # 有向图euler_path或者euler_circular_path的获取
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
        # 有向图euler_path或者euler_circular_path
        n, m = ac.read_list_ints()
        # 最关键的build_graph|
        pairs = []
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            # 每条边两个方向各走一遍
            pairs.append([u, v])
            pairs.append([v, u])

        # 有向图euler_path或者euler_circular_path的获取
        euler = DirectedEulerPath(n, pairs)
        i = euler.nodes.index(0)
        for x in euler.nodes[i:] + euler.nodes[:i]:
            ac.st(x + 1)
        return

    @staticmethod
    def lc_1743(adjacent: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/restore-the-array-from-adjacent-pairs/
        tag: undirected_euler_path|discretization
        """
        # 无向图euler_path模板题, discretization解决
        nodes = set()
        for a, b in adjacent:
            nodes.add(a)
            nodes.add(b)
        nodes = sorted(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        pairs = [[ind[x], ind[y]] for x, y in adjacent]
        n = len(nodes)
        ruler = UnDirectedEulerPath(n, pairs)
        return [nodes[x] for x in ruler.nodes]

    @staticmethod
    def lc_2097(pairs: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/valid-arrangement-of-pairs/submissions/
        tag: euler_path
        """
        # euler_path模板题，discretization后转化为图的euler_path求解
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

    @staticmethod
    def ac_4211(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4214/
        tag: directed_euler_path|specific_plan
        """
        # 有向图euler_path模板题
        n = ac.read_int()
        pairs = []
        nums = ac.read_list_ints()
        for i in range(n):
            for j in range(n):
                if j != i:
                    if nums[j] == nums[i] * 2 or nums[j] * 3 == nums[i]:
                        pairs.append([i, j])
        dt = DirectedEulerPath(n, pairs)
        ac.lst([nums[x] for x in dt.nodes])
        return