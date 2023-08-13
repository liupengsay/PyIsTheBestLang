import unittest

from typing import List

import numpy as np

from math import inf
from src.fast_io import FastIO

"""
算法：二分图最大最小权值匹配、KM算法
功能：
题目：

===================================力扣===================================
1820. 最多邀请的个数（https://leetcode.cn/problems/maximum-number-of-accepted-invitations/）使用匈牙利算法或者二分图最大权KM算法解决
1066. 校园自行车分配 II（https://leetcode.cn/problems/campus-bikes-ii/）二分图最小权KM算法解决
1947. 最大兼容性评分和（https://leetcode.cn/problems/maximum-compatibility-score-sum/）二分图最大权匹配，也可用状压DP

===================================洛谷===================================
P3386 【模板】二分图最大匹配（https://www.luogu.com.cn/problem/P3386）二分图最大匹配
P6577 【模板】二分图最大权完美匹配（https://www.luogu.com.cn/problem/P6577）二分图最大权完美匹配
P1894 [USACO4.2]完美的牛栏The Perfect Stall（https://www.luogu.com.cn/problem/P1894）二分图最大匹配，转换为网络流求解
B3605 [图论与代数结构 401] 二分图匹配（https://www.luogu.com.cn/problem/B3605）匈牙利算法二分图不带权最大匹配

================================CodeForces================================
C. Chef Monocarp（https://codeforces.com/problemset/problem/1437/C）二分图最小权匹配

================================AcWing================================
4298. 搭档（https://www.acwing.com/problem/content/4301/）匈牙利算法二分图模板题

参考：OI WiKi（xx）
"""

# EK算法
from collections import defaultdict, deque


class Hungarian:
    def __init__(self):
        # 模板：二分图不带权最大匹配
        return

    @staticmethod
    def dfs_recursion(n, m, dct):
        # 递归版写法
        assert len(dct) == m

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        # 待匹配组大小为 n
        match = [-1] * n
        ans = 0
        for x in range(m):
            # 匹配组大小为 m
            visit = [False] * n
            if hungarian(x):
                ans += 1
        return ans

    @staticmethod
    def bfs_iteration(n, m, dct):
        # 迭代版写法
        assert len(dct) == m

        match = [-1] * n
        ans = 0
        for i in range(m):
            hungarian = [0] * m
            visit = [0] * n
            stack = [[i, 0]]
            while stack:
                # 当前匹配点，与匹配对象索引
                x, ind = stack[-1]
                if ind == len(dct[x]) or hungarian[x]:
                    stack.pop()
                    continue
                y = dct[x][ind]
                if not visit[y]:
                    # 未访问过
                    visit[y] = 1
                    if match[y] == -1:
                        match[y] = x
                        hungarian[x] = 1
                    else:
                        stack.append([match[y], 0])
                else:
                    # 访问过且继任存在对应匹配
                    if hungarian[match[y]]:
                        match[y] = x
                        hungarian[x] = 1
                    stack[-1][1] += 1
            if hungarian[i]:
                ans += 1
        return ans


class EK:

    def __init__(self, n, m, s, t):
        self.flow = [0] * (n + 10)
        self.pre = [0] * (n + 10)
        self.used = set()
        self.g = defaultdict(list)
        self.edges_val = defaultdict(int)
        self.m = m
        self.s = s
        self.t = t
        self.res = 0

    def add_edge(self, from_node, to, flow):
        self.edges_val[(from_node, to)] += flow
        self.edges_val[(to, from_node)] += 0
        self.g[from_node].append(to)
        self.g[to].append(from_node)

    def bfs(self) -> bool:
        self.used.clear()
        q = deque()
        q.append(self.s)
        self.used.add(self.s)
        self.flow[self.s] = inf
        while q:
            now = q.popleft()
            for nxt in self.g[now]:
                edge = (now, nxt)
                val = self.edges_val[edge]
                if nxt not in self.used and val:
                    self.used.add(nxt)
                    self.flow[nxt] = min(self.flow[now], val)
                    self.pre[nxt] = now
                    if nxt == self.t:
                        return True
                    q.append(nxt)
        return False

    def pipline(self) -> int:
        while self.bfs():
            self.res += self.flow[self.t]
            from_node = self.t
            to = self.pre[from_node]
            while True:
                edge = (from_node, to)
                reverse_edge = (to, from_node)
                self.edges_val[edge] += self.flow[self.t]
                self.edges_val[reverse_edge] -= self.flow[self.t]
                if to == self.s:
                    break
                from_node = to
                to = self.pre[from_node]
        return self.res


class KM:
    def __init__(self):
        self.matrix = None
        self.max_weight = 0
        self.row, self.col = 0, 0  # 源数据行列
        self.size = 0   # 方阵大小
        self.lx = None  # 左侧权值
        self.ly = None  # 右侧权值
        self.match = None   # 匹配结果
        self.slack = None   # 边权和顶标最小的差值
        self.visx = None    # 左侧是否加入增广路
        self.visy = None    # 右侧是否加入增广路

    # 调整数据
    def pad_matrix(self, min):
        if min:
            max = self.matrix.max() + 1
            self.matrix = max-self.matrix

        if self.row > self.col:   # 行大于列，添加列
            self.matrix = np.c_[self.matrix, np.array([[0] * (self.row - self.col)] * self.row)]
        elif self.col > self.row:  # 列大于行，添加行
            self.matrix = np.r_[self.matrix, np.array([[0] * self.col] * (self.col - self.row))]

    def reset_slack(self):
        self.slack.fill(self.max_weight + 1)

    def reset_vis(self):
        self.visx.fill(False)
        self.visy.fill(False)

    def find_path(self, x):
        self.visx[x] = True
        for y in range(self.size):
            if self.visy[y]:
                continue
            tmp_delta = self.lx[x] + self.ly[y] - self.matrix[x][y]
            if tmp_delta == 0:
                self.visy[y] = True
                if self.match[y] == -1 or self.find_path(self.match[y]):
                    self.match[y] = x
                    return True
            elif self.slack[y] > tmp_delta:
                self.slack[y] = tmp_delta

        return False

    def km_cal(self):
        for x in range(self.size):
            self.reset_slack()
            while True:
                self.reset_vis()
                if self.find_path(x):
                    break
                else:  # update slack
                    delta = self.slack[~self.visy].min()
                    self.lx[self.visx] -= delta
                    self.ly[self.visy] += delta
                    self.slack[~self.visy] -= delta

    def compute(self, datas, min=False):
        """
        :param datas: 权值矩阵
        :param min: 是否取最小组合，默认最大组合
        :return: 输出行对应的结果位置
        """
        self.matrix = np.array(datas) if not isinstance(datas, np.ndarray) else datas
        self.max_weight = self.matrix.sum()
        self.row, self.col = self.matrix.shape  # 源数据行列
        self.size = max(self.row, self.col)
        self.pad_matrix(min)
        self.lx = self.matrix.max(1)
        self.ly = np.array([0] * self.size, dtype=int)
        self.match = np.array([-1] * self.size, dtype=int)
        self.slack = np.array([0] * self.size, dtype=int)
        self.visx = np.array([False] * self.size, dtype=bool)
        self.visy = np.array([False] * self.size, dtype=bool)

        self.km_cal()

        match = [i[0] for i in sorted(enumerate(self.match), key=lambda x: x[1])]
        result = []
        for i in range(self.row):
            result.append((i, match[i] if match[i] < self.col else -1))  # 没有对应的值给-1
        return result


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1820(grid):
        # 模板：匈牙利算法模板建图计算最大匹配
        m, n = len(grid), len(grid[0])
        dct = defaultdict(list)
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    dct[i].append(j)

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        match = [-1] * n
        ans = 0
        for i in range(m):
            visit = [False]*n
            if hungarian(i):
                ans += 1
        return ans

    @staticmethod
    def lc_1820_2(grid: List[List[int]]) -> int:
        # 模板：EK网络最大流算法模板建图计算最大匹配
        n = len(grid)
        m = len(grid[0])
        s = n + m + 1
        t = n + m + 2
        ek = EK(n + m, n * m, s, t)
        for i in range(n):
            for j in range(m):
                if grid[i][j]:
                    ek.add_edge(i, n + j, 1)
        for i in range(n):
            ek.add_edge(s, i, 1)
        for i in range(m):
            ek.add_edge(n + i, t, 1)
        return ek.pipline()
    
    @staticmethod
    def lc_1820_3(grid):
        # 模板：KM算法模板建图计算最大匹配
        n = max(len(grid), len(grid[0]))
        lst = [[0]*n for _ in range(n)]
        ind = 0
        for i in range(n):
            for j in range(n):
                try:
                    lst[i][j] = grid[i][j]
                except IndexError as _:
                    ind += 1

        arr = np.array(lst)
        km = KM()
        max_ = km.compute(arr)

        ans = 0
        for i, j in max_:
            ans += lst[i][j]
        return ans

    @staticmethod
    def lg_p1894(ac=FastIO()):
        # 模板：二分图最大权匹配（不带权也可以使用匈牙利算法）
        n, m = ac.read_ints()
        s = n + m + 1
        t = n + m + 2
        # 集合个数n与集合个数m
        ek = EK(n + m, n * m, s, t)
        for i in range(n):
            lst = ac.read_list_ints()[1:]
            for j in lst:
                # 增加边
                ek.add_edge(i, n + j - 1, 1)
        # 增加超级源点与汇点
        for i in range(n):
            ek.add_edge(s, i, 1)
        for i in range(m):
            ek.add_edge(n + i, t, 1)
        ac.st(ek.pipline())
        return

    @staticmethod
    def lg_3386(ac=FastIO()):
        # 模板：匈牙利算法二分图不带权最大匹配
        n, m, e = ac.read_ints()
        dct = [[] for _ in range(m)]
        for _ in range(e):
            i, j = ac.read_ints()
            i -= 1
            j -= 1
            dct[j].append(i)
        ac.st(Hungarian().dfs_recursion(n, m, dct))
        # ac.st(Hungarian().bfs_iteration(n, m, dct))
        return

    @staticmethod
    def lc_1066(workers: List[List[int]], bikes: List[List[int]]) -> int:
        # 模板：二分图最小权匹配
        n = len(workers)
        m = len(bikes)
        grid = [[0]*m for _ in range(m)]
        for i in range(n):
            for j in range(m):
                grid[i][j] = abs(workers[i][0]-bikes[j][0])+abs(workers[i][1]-bikes[j][1])

        a = np.array(grid)
        km = KM()
        min_ = km.compute(a.copy(), min=True)
        ans = 0
        for i,j in min_:
            ans += grid[i][j]
        return ans

    @staticmethod
    def ac_4298(ac=FastIO()):
        # 模板：匈牙利算法二分图模板题
        m = ac.read_int()
        a = ac.read_list_ints()
        n = ac.read_int()
        b = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if abs(a[i]-b[j]) <= 1:
                    dct[i].append(j)
        ans = Hungarian().dfs_recursion(n, m, dct)
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_km(self):
        a = np.array([[1, 3, 5], [4, 1, 1], [1, 5, 3]])

        km = KM()
        min_ = km.compute(a.copy(), True)
        print("最小组合:", min_,  a[[i[0] for i in min_], [i[1] for i in min_]])

        max_ = km.compute(a.copy())
        print("最大组合:", max_, a[[i[0] for i in max_], [i[1] for i in max_]])
        return


if __name__ == '__main__':
    unittest.main()
