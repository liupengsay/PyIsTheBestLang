import decimal
import math
import unittest

from typing import List
from collections import defaultdict, Counter
from algorithm.src.fast_io import FastIO
import heapq

from algorithm.src.graph.dijkstra import Dijkstra
from algorithm.src.mathmatics.number_theory import NumberTheory
from math import inf


"""

算法：并查集、可持久化并查集
功能：用来处理图论相关的联通问题，通常结合逆向思考、置换环或者离线查询进行求解，连通块不一定是秩大小，也可以是最大最小值、和等
题目：

===================================力扣===================================
1697. 检查边长度限制的路径是否存在（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/）排序后离线查询两点间所有路径的最大边权值
2503. 矩阵查询可获得的最大分数（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/）排序后离线查询与起点相连的连通块的大小
2421. 好路径的数目（https://leetcode.cn/problems/number-of-good-paths/）根据权值进行排序更新并查集计算连通分块满足条件的节点对数
2382. 删除操作后的最大子段和（https://leetcode.cn/problems/maximum-segment-sum-after-removals/）逆向进行访问查询并更新连通块的结果
2334. 元素值大于变化阈值的子数组（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）排序后枚举动态维护并查集连通块
2158. 每天绘制新区域的数量（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）使用并查集维护区间左端点，不断进行合并
2157. 字符串分组（https://leetcode.cn/problems/groups-of-strings/）利用字母的有限数量进行变换枚举分组
2076. 处理含限制条件的好友请求（https://leetcode.cn/problems/process-restricted-friend-requests/）使用并查集变种，维护群体的不喜欢关系
2459. 通过移动项目到空白区域来排序数组（https://leetcode.cn/problems/sort-array-by-moving-items-to-empty-space/）置换环经典题目


===================================洛谷===================================
P3367 并查集（https://www.luogu.com.cn/problem/P3367）计算连通分块的数量
P5836 Milk Visits S（https://www.luogu.com.cn/problem/P5836）使用两个并查集进行不同方面的查询
P3144 [USACO16OPEN]Closing the Farm S（https://www.luogu.com.cn/problem/P3144）逆序并查集，考察连通块的数量
P5836 [USACO19DEC]Milk Visits S（https://www.luogu.com.cn/problem/P5836）两个并查集进行连通情况查询
P5877 棋盘游戏（https://www.luogu.com.cn/problem/P5877）正向模拟实时计算更新连通块的数量
P6111 [USACO18JAN]MooTube S（https://www.luogu.com.cn/problem/P6111）并查集加离线查询进行计算
P6121 [USACO16OPEN]Closing the Farm G（https://www.luogu.com.cn/problem/P6121）逆序并查集根据连通块大小进行连通性判定
P6153 询问（https://www.luogu.com.cn/problem/P6153）经典并查集思想贪心题，体现了并查集的思想
P1955 [NOI2015] 程序自动分析（https://www.luogu.com.cn/problem/P1955）并查集裸题
P1196 [NOI2002] 银河英雄传说（https://www.luogu.com.cn/problem/P1196）带权并查集
P1197 [JSOI2008] 星球大战（https://www.luogu.com.cn/problem/P1197）逆序并查集，倒序枚举计算联通块个数
P1522 [USACO2.4] 牛的旅行 Cow Tours（https://www.luogu.com.cn/problem/P1522）连通块，枚举新增路径并高精度计算联通块直径
P1621 集合（https://www.luogu.com.cn/problem/P1621）利用素数筛的思想对数复杂度合并公共质因数大于p的数并计算连通块数量
P1892 [BOI2003]团伙（https://www.luogu.com.cn/problem/P1892）经典并查集，敌人与朋友关系
P2189 小Z的传感器（https://www.luogu.com.cn/problem/P2189）并查集经典题，确定访问顺序的合法性
P2307 迷宫（https://www.luogu.com.cn/problem/P2307）并查集判定树的生成是否合法

================================CodeForces================================
D. Roads not only in Berland（https://codeforces.com/problemset/problem/25/D）并查集将原来的边断掉重新来连接使得成为一整个连通集
E. Monsters（https://codeforces.com/contest/1810/problem/E）并查集加启发式搜索，使用BFS与堆优化实现
E. Connected Components?（https://codeforces.com/contest/920/problem/E）并查集，加线性动态维护剩余节点
C. Ice Cave（https://codeforces.com/problemset/problem/540/C）路径可达
E2. Unforgivable Curse (hard version)（https://codeforces.com/problemset/problem/1800/E2）使用并查集分组计算可达



参考：OI WiKi（xx）
"""


# 标准并查集
class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class UnionFindWeighted:
    def __init__(self, n: int) -> None:
        # 模板：带权并查集
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        self.front = [0]*n  # 离队头的距离
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        lst.append(x)
        m = len(lst)
        for i in range(m-2, -1, -1):
            self.front[lst[i]] += self.front[lst[i+1]]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        # 将 root_x 拼接到 root_y 后面
        self.front[root_x] += self.size[root_y]
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


# 可持久化并查集
class PersistentUnionFind:
    def __init__(self, n):
        self.rank = [0] * n
        self.root = list(range(n))
        self.version = [float("inf")] * n

    def union(self, x, y, tm):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.version[root_y] = tm
                self.root[root_y] = root_x
            else:
                self.version[root_x] = tm
                self.root[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
            return True
        return False

    def find(self, x, tm=float("inf")):
        if x == self.root[x] or self.version[x] >= tm:
            return x
        return self.find(self.root[x], tm)

    def is_connected(self, x, y, tm):
        return self.find(x, tm) == self.find(y, tm)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1810e(ac=FastIO()):
        # 模板：并查集加启发式搜索，使用线性遍历维护
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
            nums = ac.read_list_ints()
            edge = [[] for _ in range(n)]
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                edge[u].append(v)
                edge[v].append(u)

            visit = [-1] * n
            ans = "NO"
            for i in range(n):
                if visit[i] == -1 and not nums[i]:
                    count = 0
                    visit[i] = i
                    stack = [[0, i]]
                    while stack:
                        d, x = heapq.heappop(stack)
                        if count < nums[x]:
                            break
                        count += 1
                        for j in edge[x]:
                            if visit[j] != i:
                                visit[j] = i
                                heapq.heappush(stack, [nums[j], j])
                    if count == n:
                        ans = "YES"
                        break
            ac.st(ans)

        return

    @staticmethod
    def cf_920e(ac=FastIO()):
        # 模板：并查集线性更新，使用集合进行维护
        n, m = ac.read_list_ints()
        edge = set()
        for _ in range(m):
            u, v = ac.read_ints_minus_one()
            edge.add((u, v))
        ans = []
        not_visit = set(range(n))
        for i in range(n):
            if i in not_visit:
                stack = [i]
                cnt = 1
                not_visit.discard(i)
                while stack:
                    u = stack.pop()
                    visit = []
                    for v in not_visit:
                        if (u, v) in edge or (v, u) in edge:
                            continue
                        cnt += 1
                        stack.append(v)
                        visit.append(v)
                    for v in visit:
                        not_visit.discard(v)
                ans.append(cnt)
        ans.sort()
        ac.st(len(ans))
        ac.lst(ans)
        return


    @staticmethod
    def lc_1697(n: int, edge_list: List[List[int]], queries: List[List[int]]) -> List[bool]:
        # 模板：并查集与离线排序查询结合
        m = len(queries)

        # 按照 limit 排序
        ind = list(range(m))
        ind.sort(key=lambda x: queries[x][2])

        # 按照边权值排序
        edge_list.sort(key=lambda x: x[2])
        uf = UnionFind(n)
        i = 0
        k = len(edge_list)
        ans = []
        # 查询 queries 里面的 [p, q, limit] 即 p 和 q 之间存在最大边权值严格小于 limit 的路径是否成立
        for j in ind:
            # 实时加入可用于连通的边并查询结果
            p, q, limit = queries[j]
            while i < k and edge_list[i][2] < limit:
                uf.union(edge_list[i][0], edge_list[i][1])
                i += 1
            ans.append([j, uf.is_connected(p, q)])

        # 按照顺序返回结果
        ans.sort(key=lambda x: x[0])
        return [an[1] for an in ans]

    @staticmethod
    def lc_2503(grid: List[List[int]], queries: List[int]) -> List[int]:
        # 模板：并查集与离线排序查询结合
        dct = []
        # 根据邻居关系进行建图处理
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    x, y = grid[i][j], grid[i + 1][j]
                    dct.append([i * n + j, i * n + n + j, x if x > y else y])
                if j + 1 < n:
                    x, y = grid[i][j], grid[i][j + 1]
                    dct.append([i * n + j, i * n + 1 + j, x if x > y else y])
        dct.sort(key=lambda d: d[2])
        uf = UnionFind(m * n)

        # 按照查询值的大小排序，依次进行查询
        k = len(queries)
        ind = list(range(k))
        ind.sort(key=lambda d: queries[d])

        # 根据查询值的大小利用指针持续更新并查集
        ans = [0]*k
        j = 0
        length = len(dct)
        for i in ind:
            cur = queries[i]
            while j < length and dct[j][2] < cur:
                uf.union(dct[j][0], dct[j][1])
                j += 1
            if cur > grid[0][0]:
                ans[i] = uf.size[uf.find(0)]
        return ans

    @staticmethod
    def lc_2421(vals: List[int], edges: List[List[int]]) -> int:
        # 模板：并查集与离线排序查询结合
        n = len(vals)
        index = defaultdict(list)
        for i in range(n):
            index[vals[i]].append(i)
        edges.sort(key=lambda x: max(vals[x[0]], vals[x[1]]))
        uf = UnionFind(n)
        # 离线查询计数
        i = 0
        m = len(edges)
        ans = 0
        for val in sorted(index):
            while i < m and vals[edges[i][0]] <= val and vals[edges[i][1]] <= val:
                uf.union(edges[i][0], edges[i][1])
                i += 1
            cnt = Counter(uf.find(x) for x in index[val])
            for w in cnt.values():
                ans += w * (w - 1) // 2 + w
        return ans

    @staticmethod
    def lg_p1196(ac=FastIO()):
        # 模板：计算带权并查集
        uf = UnionFindWeighted(30000)
        for _ in range(ac.read_int()):
            lst = ac.read_list_strs()
            i, j = [int(w) - 1 for w in lst[1:]]
            if lst[0] == "M":
                uf.union(i, j)
            else:
                root_x = uf.find(i)
                root_y = uf.find(j)
                if root_x != root_y:
                    ac.st(-1)
                else:
                    ac.st(abs(uf.front[i]-uf.front[j])-1)
        return

    @staticmethod
    def lg_p1197(ac=FastIO()):
        # 模板：逆序并查集，倒序枚举计算联通块个数
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints()
            dct[i].append(j)
            dct[j].append(i)
        k = ac.read_int()
        rem = [ac.read_int() for _ in range(k)]
        out = set(rem)
        uf = UnionFind(n)
        for i in range(n):
            if i not in out:
                for j in dct[i]:
                    if j not in out:
                        uf.union(i, j)
        ans = []
        for i in range(k-1, -1, -1):
            ans.append(uf.part-i-1)
            out.discard(rem[i])
            for j in dct[rem[i]]:
                if j not in out:
                    uf.union(rem[i], j)
        ans.append(uf.part)
        ans.reverse()
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p1522(ac=FastIO()):

        # 模板：连通块，枚举新增路径并高精度计算联通块直径

        def dis(x1, y1, x2, y2):
            return math.sqrt(decimal.Decimal(((x1 - x2) ** 2 + (y1 - y2) ** 2)))

        n = ac.read_int()
        nums = [[w for w in ac.read_list_ints()] for _ in range(n)]
        grid = [ac.read_str() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if grid[i][j] == "1":
                    uf.union(i, j)
                    d = dis(nums[i][0], nums[i][1], nums[j][0], nums[j][1])
                    dct[i][j] = dct[j][i] = d

        dist = []
        for i in range(n):
            dist.append(Dijkstra().get_dijkstra_result(dct, i))

        part = uf.get_root_part()
        fast = [inf] * n
        group = dict()
        for p in part:
            for i in part[p]:
                fast[i] = max(dist[i][j] for j in part[p])
            group[p] = max(fast[i] for i in part[p])

        ans = inf
        for i in range(n):
            for j in range(n):
                if not uf.is_connected(i, j):
                    cur = dis(nums[i][0], nums[i][1], nums[j][0], nums[j][1]) + fast[i] + fast[j]
                    cur = ac.max(cur, group[uf.find(i)])
                    cur = ac.max(cur, group[uf.find(j)])
                    ans = ac.min(ans, cur)
        ac.st("%.6f" % ans)
        return

    @staticmethod
    def lg_p1621(ac=FastIO()):
        # 模板：利用素数筛的思想对数复杂度合并公共质因数大于p的数并计算连通块数量
        a, b, p = ac.read_list_ints()
        nums = list(range(a, b + 1))
        ind = {num: num - a for num in nums}
        primes = [x for x in NumberTheory().sieve_of_eratosthenes(b) if x >= p]

        # 利用素数进行合并
        uf = UnionFind(b - a + 1)
        for x in primes:
            lst = []
            y = x
            while y <= b:
                if y in ind:
                    lst.append(ind[y])
                y += x
            m = len(lst)
            for j in range(m - 1):
                uf.union(lst[j], lst[j + 1])
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p1892(ac=FastIO()):
        # 模板：经典并查集，敌人与朋友关系
        n = ac.read_int()
        m = ac.read_int()
        uf = UnionFind(n)
        dct = dict()
        for _ in range(m):
            lst = [w for w in input().strip().split() if w]
            a, b = int(lst[1]), int(lst[2])
            a -= 1
            b -= 1
            if lst[0] == "E":
                # 敌人的敌人是朋友
                if a in dct:
                    uf.union(dct[a], b)
                if b in dct:
                    uf.union(dct[b], a)
                dct[a] = b
                dct[b] = a
            else:
                uf.union(a, b)
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p1955(ac=FastIO()):
        # 模板：并查集裸题
        t = ac.read_int()
        for _ in range(t):
            n = ac.read_int()
            ind = dict()
            uf = UnionFind(n * 2)
            res = []
            for _ in range(n):
                lst = ac.read_list_ints()
                while not lst:
                    lst = ac.read_list_ints()
                i, j, e = lst
                if i not in ind:
                    ind[i] = len(ind)
                if j not in ind:
                    ind[j] = len(ind)
                if e == 1:
                    uf.union(ind[i], ind[j])
                else:
                    res.append([ind[i], ind[j]])
            if any(uf.is_connected(i, j) for i, j in res):
                ac.st("NO")
            else:
                ac.st("YES")
        return

    @staticmethod
    def lg_p2189(ac=FastIO()):

        # 模板：并查集经典题，确定访问顺序的合法性
        n, m, k, q = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        for _ in range(q):
            order = ac.read_list_ints_minus_one()
            uf = UnionFind(n)
            visit = [0] * n
            for i in order:
                visit[i] = 1

            # 不在路径上的直接连通
            ans = True
            pre = order[0]
            for i in range(n):
                if not visit[i]:
                    for j in dct[i]:
                        if not visit[j]:
                            uf.union(i, j)

            # 遍历连接确认当前的连通性
            for i in order:
                visit[i] = 0
                for j in dct[i]:
                    if not visit[j]:
                        uf.union(i, j)
                if not uf.is_connected(i, pre):
                    ans = False
                    break
                pre = i
            ac.st("Yes" if ans else "No")
        return

    @staticmethod
    def lg_p2307(ac=FastIO()):
        # 模板：并查集判定树的生成是否合法
        while True:
            ans = []
            while True:
                lst = ac.read_list_ints()
                if not lst:
                    break
                ans.extend(lst)
            if ans == [-1, -1]:
                break
            ans = ans[:-2]
            nodes = list(set(ans))
            ind = {num: i for i, num in enumerate(nodes)}
            m = len(ind)
            uf = UnionFind(m)
            res = True
            for i in range(0, len(ans), 2):
                a, b = ind[ans[i]], ind[ans[i + 1]]
                if not uf.union(a, b):
                    res = False
                    break
            ac.st(1 if res and uf.part == 1 else 0)
        return


class TestGeneral(unittest.TestCase):

    def test_union_find(self):
        uf = UnionFind(5)
        for i, j in [[0, 1], [1, 2]]:
            uf.union(i, j)
        assert uf.part == 3
        return

    def test_solution(self):
        # 离线根据时间戳排序进行查询
        sl = Solution()
        n = 3
        edge_list = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        queries = [[0, 1, 2], [0, 2, 5]]
        assert sl.distance_limited_paths_exist(n, edge_list, queries) == [False, True]

    def test_persistent_union_find(self):
        # 在线根据历史版本时间戳查询
        n = 3
        puf = PersistentUnionFind(n)
        edge_list = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        edge_list.sort(key=lambda item: item[2])
        for x, y, tm in edge_list:
            puf.union(x, y, tm)
        queries = [[0, 1, 2], [0, 2, 5]]
        assert [puf.is_connected(x, y, tm) for x, y, tm in queries] == [False, True]


if __name__ == '__main__':
    unittest.main()
