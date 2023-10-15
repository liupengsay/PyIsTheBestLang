import unittest

from typing import List

import numpy as np

from math import inf
from src.fast_io import FastIO

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


