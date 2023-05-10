import random
import unittest
from collections import defaultdict, deque
from types import GeneratorType
from typing import List

from algorithm.src.fast_io import inf, FastIO

"""
算法：线段树
功能：用以修改和查询区间的值信息，支持增减、修改，区间和、区间最大值、区间最小值
题目：

===================================力扣===================================
218. 天际线问题（https://leetcode.cn/problems/the-skyline-problem/solution/by-liupengsay-isfo/）区间值修改与计算最大值
2286. 以组为单位订音乐会的门票（https://leetcode.cn/problems/booking-concert-tickets-in-groups/）区间值增减与计算区间和、区间最大值、区间最小值
2407. 最长递增子序列 II（https://leetcode.cn/problems/longest-increasing-subsequence-ii/）维护与查询区间最大值，然后进行DP更新
2276. 统计区间中的整数数目（https://leetcode.cn/problems/count-integers-in-intervals/）维护区间并集的长度
2179. 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2158. 每天绘制新区域的数量（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）线段树维护区间范围的覆盖
6358. 更新数组后处理求和查询（https://leetcode.cn/problems/handling-sum-queries-after-update/）区间值01翻转与区间和查询，使用bitset实现
6318. 完成所有任务的最少时间（https://leetcode.cn/contest/weekly-contest-336/problems/minimum-time-to-complete-all-tasks/）线段树，贪心加二分


===================================洛谷===================================
P2846 [USACO08NOV]Light Switching G（https://www.luogu.com.cn/problem/P2846）线段树统计区间翻转和
P2574 XOR的艺术（https://www.luogu.com.cn/problem/P2574）线段树统计区间翻转和
P3130 [USACO15DEC] Counting Haybale P（https://www.luogu.com.cn/problem/P3130）区间增减、区间最小值查询、区间和查询
P3870 [TJOI2009] 开关（https://www.luogu.com.cn/problem/P3870） 区间值01翻转与区间和查询
P5057 [CQOI2006] 简单题（https://www.luogu.com.cn/problem/P5057） 区间值01翻转与区间和查询
P3372 【模板】线段树 1（https://www.luogu.com.cn/problem/P3372）区间值增减与求和
P2880 [USACO07JAN] Balanced Lineup G（https://www.luogu.com.cn/problem/P2880）查询区间最大值与最小值
P1904 天际线（https://www.luogu.com.cn/problem/P1904）使用线段树，区间更新最大值并单点查询计算天际线
P1438 无聊的数列（https://www.luogu.com.cn/problem/P1438）差分数组区间增减加线段树查询区间和
P1253 扶苏的问题（https://www.luogu.com.cn/problem/P1253）区间增减与区间修改并使用线段树查询区间和
P3373 【模板】线段树 2（https://www.luogu.com.cn/problem/P3373）区间乘法与区间加法并使用线段树查询区间和
P4513 小白逛公园（https://www.luogu.com.cn/problem/P4513）单点修改与区间最大连续子数组和查询，可升级为区间修改
P1471 方差（https://www.luogu.com.cn/problem/P1471）区间增减，维护区间和与区间数字平方的和，以计算均差与方差
P6492 [COCI2010-2011#6] STEP（https://www.luogu.com.cn/problem/P6492）单点修改，查找最长的01交替字符子串连续区间
P4145 上帝造题的七分钟 2 / 花神游历各国（https://www.luogu.com.cn/problem/P4145）区间值开方向下取整，区间和查询
P1558 色板游戏（https://www.luogu.com.cn/problem/P1558）线段树区间值修改，区间或值查询

================================CodeForces================================

https://codeforces.com/problemset/problem/482/B（区间按位或赋值、按位与查询）
C. Sereja and Brackets（https://codeforces.com/problemset/problem/380/C）线段树查询区间内所有合法连续子序列括号串的总长度
C. Circular RMQ（https://codeforces.com/problemset/problem/52/C）线段树更新和查询循环数组区间最小值
D. The Child and Sequence（https://codeforces.com/problemset/problem/438/D）使用线段树维护区间取模，区间和，修改单点值，和区间最大值
E. A Simple Task（https://codeforces.com/contest/558/problem/E）26个线段树维护区间排序信息
D. Water Tree（https://codeforces.com/problemset/problem/343/D）dfs序加线段树
E. XOR on Segment（https://codeforces.com/problemset/problem/242/E）线段树区间异或，与区间加和

参考：OI WiKi（xx）
"""


class SegBitSet:
    # 使用位运算进行区间01翻转操作
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # 索引从0开始
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        p = (1 << (c + 1)) - (1 << b)
        return (self.val & p).bit_count()


class SegTreeBrackets:
    def __init__(self, n, s):
        self.n = n
        self.s = s
        self.a = [0] * (2 * self.n)
        self.b = [0] * (2 * self.n)
        self.c = [0] * (2 * self.n)

    def build(self):
        for i in range(self.n):
            self.a[i + self.n] = 0
            self.b[i + self.n] = 1 if self.s[i] == "(" else 0
            self.c[i + self.n] = 1 if self.s[i] == ")" else 0
        for i in range(self.n - 1, 0, -1):
            t = min(self.b[i << 1], self.c[i << 1 | 1])
            self.a[i] = self.a[i << 1] + self.a[i << 1 | 1] + 2 * t
            self.b[i] = self.b[i << 1] + self.b[i << 1 | 1] - t
            self.c[i] = self.c[i << 1] + self.c[i << 1 | 1] - t

    def query(self, low, r):
        left = []
        right = []
        low += self.n
        r += self.n
        while low <= r:
            if low & 1:
                left.append([self.a[low], self.b[low], self.c[low]])
                low += 1
            if not r & 1:
                right.append([self.a[r], self.b[r], self.c[r]])
                r -= 1
            low >>= 1
            r >>= 1
        a1 = b1 = c1 = 0
        for a2, b2, c2 in left + right[::-1]:
            t = min(b1, c2)
            a1 += a2 + 2 * t
            b1 += b2 - t
            c1 += c2 - t
        return a1


class SegmentTreeRangeAddMax:
    # 模板：线段树区间更新、持续增加最大值
    def __init__(self, n):
        self.floor = 0
        self.height = [self.floor]*(4*n)
        self.lazy = [self.floor]*(4*n)

    @staticmethod
    def max(a, b):
        return a if a > b else b

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.max(self.height[2 * i], self.lazy[i])
            self.height[2 * i + 1] = self.max(self.height[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self.max(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self.max(self.lazy[2 * i + 1], self.lazy[i])
            self.lazy[i] = self.floor
        return

    def update(self, left, right, s, t, val, i):
        # 更新区间最大值
        stack = [[s, t, i, 1]]
        while stack:
            a, b, i, state = stack.pop()
            if state:
                if left <= a and b <= right:
                    self.height[i] = self.max(self.height[i], val)
                    self.lazy[i] = self.max(self.lazy[i], val)
                    continue
                self.push_down(i)
                stack.append([a, b, i, 0])
                m = a + (b - a) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([a, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1, 1])
            else:
                self.height[i] = self.max(self.height[2 * i], self.height[2 * i + 1])
        return

    def query(self, left, right, s, t, i):
        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = self.floor
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                highest = self.max(highest, self.height[i])
                continue
            self.push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2*i])
            if right > m:
                stack.append([m+1, b, 2*i + 1])
        return highest


class SegmentTreeRangeAddMin:
    # 模板：线段树区间更新、持续减小最小值
    def __init__(self, n):
        self.ceil = float("inf")
        self.height = [self.ceil]*(4*n)
        self.lazy = [self.ceil]*(4*n)

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def push_down(self, i):
        # 懒标记下放，注意取最小值
        if self.lazy[i] != self.ceil:
            self.height[2 * i] = self.min(self.height[2 * i], self.lazy[i])
            self.height[2 * i + 1] = self.min(self.height[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self.min(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self.min(self.lazy[2 * i + 1], self.lazy[i])

            self.lazy[i] = self.ceil
        return

    def update(self, left, right, s, t, val, i):
        # 更新区间最小值
        stack = [[s, t, i, 1]]
        while stack:
            a, b, i, state = stack.pop()
            if state:
                if left <= a and b <= right:
                    self.height[i] = self.min(self.height[i], val)
                    self.lazy[i] = self.min(self.lazy[i], val)
                    continue
                self.push_down(i)
                stack.append([a, b, i, 0])
                m = a + (b - a) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([a, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1, 1])
            else:
                self.height[i] = self.min(self.height[2 * i], self.height[2 * i + 1])
        return

    def query(self, left, right, s, t, i):
        # 查询区间的最小值
        stack = [[s, t, i]]
        floor = self.ceil
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                floor = self.min(floor, self.height[i])
                continue
            self.push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2*i])
            if right > m:
                stack.append([m+1, b, 2*i + 1])
        return floor


class SegmentTreeRangeUpdateQuerySumMinMax:
    def __init__(self, nums: List[int]) -> None:
        # 模板：区间值增减、区间和查询、区间最小值查询、区间最大值查询
        self.inf = float("inf")
        self.n = len(nums)
        self.nums = nums
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记
        self.floor = [0] * (4 * self.n)  # 最小值
        self.ceil = [0] * (4 * self.n)  # 最大值
        self.build()  # 初始化线段树
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, ind, state = stack.pop()
            if state:
                if s == t:
                    self.cover[ind] = self.nums[s]
                    self.ceil[ind] = self.nums[s]
                    self.floor[ind] = self.nums[s]
                else:
                    stack.append([s, t, ind, 0])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind, 1])
                    stack.append([m + 1, t, 2 * ind + 1, 1])
            else:
                self.cover[ind] = self.cover[2 * ind] + self.cover[2 * ind + 1]
                self.ceil[ind] = self.max(self.ceil[2 * ind], self.ceil[2 * ind + 1])
                self.floor[ind] = self.min(self.floor[2 * ind], self.floor[2 * ind + 1])
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] += self.lazy[i] * (t - m)

            self.floor[2 * i] += self.lazy[i]
            self.floor[2 * i + 1] += self.lazy[i]

            self.ceil[2 * i] += self.lazy[i]
            self.ceil[2 * i + 1] += self.lazy[i]

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.cover[i] += val * (t - s + 1)
                    self.floor[i] += val
                    self.ceil[i] += val
                    self.lazy[i] += val
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
                self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
                self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans

    def query_min(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的最小值
        stack = [[s, t, i]]
        highest = self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest

    def query_max(self, left: int, right: int, s: int, t: int, i: int) -> int:

        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = -self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest


class SegmentTreeRangeChangeQuerySumMinMax:
    def __init__(self, nums):
        # 模板：区间值修改、区间和查询、区间最小值查询、区间最大值查询
        self.inf = float("inf")
        self.n = len(nums)
        self.nums = nums
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [self.inf] * (4 * self.n)  # 懒标记
        self.floor = [0] * (4 * self.n)  # 最小值
        self.ceil = [0] * (4 * self.n)  # 最大值
        self.build()  # 初始化数组

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def build(self):

        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, ind, state = stack.pop()
            if state:
                if s == t:
                    self.cover[ind] = self.nums[s]
                    self.ceil[ind] = self.nums[s]
                    self.floor[ind] = self.nums[s]
                else:
                    stack.append([s, t, ind, 0])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind, 1])
                    stack.append([m + 1, t, 2 * ind + 1, 1])
            else:
                self.cover[ind] = self.cover[2 * ind] + self.cover[2 * ind + 1]
                self.ceil[ind] = self.max(self.ceil[2 * ind], self.ceil[2 * ind + 1])
                self.floor[ind] = self.min(self.floor[2 * ind], self.floor[2 * ind + 1])
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i] != self.inf:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.floor[2 * i] = self.lazy[i]
            self.floor[2 * i + 1] = self.lazy[i]

            self.ceil[2 * i] = self.lazy[i]
            self.ceil[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = self.inf


    def change(self, left, right, s, t, val, i):
        # 更新区间值
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.cover[i] = val * (t - s + 1)
                    self.floor[i] = val
                    self.ceil[i] = val
                    self.lazy[i] = val
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
                self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
                self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def query_sum(self, left, right, s, t, i):
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans

    def query_min(self, left, right, s, t, i):
        # 查询区间的最小值
        stack = [[s, t, i]]
        highest = self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest

    def query_max(self, left, right, s, t, i):

        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = -self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest


class SegmentTreeRangeUpdateChangeQueryMax:
    def __init__(self, nums: List[int]) -> None:
        # 模板：区间值增减、区间值修改、区间最大值查询
        self.inf = float("inf")
        self.n = len(nums)
        self.nums = nums
        self.lazy = [[self.inf, 0]] * (4 * self.n)  # 懒标记
        self.ceil = [-self.inf] * (4 * self.n)  # 最大值
        self.build()  # 初始化线段树
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, ind, state = stack.pop()
            if state:
                if s == t:
                    self.ceil[ind] = self.nums[s]
                else:
                    stack.append([s, t, ind, 0])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind, 1])
                    stack.append([m + 1, t, 2 * ind + 1, 1])
            else:
                self.ceil[ind] = self.max(self.ceil[2 * ind], self.ceil[2 * ind + 1])
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i] != [self.inf, 0]:
            a, b = self.lazy[i]  # 分别表示修改为 a 与 增加 b
            if a == self.inf:
                self.ceil[2 * i] += b
                self.ceil[2 * i + 1] += b
                self.lazy[2 * i] = [self.inf, self.lazy[2 * i][1] + b]
                self.lazy[2 * i + 1] = [self.inf, self.lazy[2 * i + 1][1] + b]
            else:
                self.ceil[2 * i] = a
                self.ceil[2 * i + 1] = a
                self.lazy[2 * i] = [a, 0]
                self.lazy[2 * i + 1] = [a, 0]
            self.lazy[i] = [self.inf, 0]

    def update(self, left: int, right: int, s: int, t: int, val: int, flag: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    if flag == 1:
                        self.ceil[i] = val
                        self.lazy[i] = [val, 0]
                    elif self.lazy[i][0] != self.inf:
                        self.ceil[i] += val
                        self.lazy[i] = [self.lazy[i][0]+val, 0]
                    else:
                        self.ceil[i] += val
                        self.lazy[i] = [self.inf, self.lazy[i][1]+val]
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int) -> int:

        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = -self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest



class SegmentTreeOrUpdateAndQuery:
    def __init__(self):
        # 模板：区间按位或赋值、按位与查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i):
        if self.lazy[i]:
            self.cover[2 * i] |= self.lazy[i]
            self.cover[2 * i + 1] |= self.lazy[i]

            self.lazy[2 * i] |= self.lazy[i]
            self.lazy[2 * i + 1] |= self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] |= val
            self.lazy[i] |= val
            return
        m = s + (t - s) // 2
        self.push_down(i)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] & self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i)
        ans = (1 << 31) - 1
        if left <= m:
            ans &= self.query(left, r, s, m, 2 * i)
        if r > m:
            ans &= self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreeRangeUpdateXORSum:
    def __init__(self, n):
        # 模板：区间值01翻转与区间和查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2 * i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意使用异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意使用异或抵消查询

            self.lazy[i] = 0
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy[i] ^= val  # 注意使用异或抵消查询
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class SegmentTreeRangeAddSum:
    def __init__(self):
        # 模板：区间值增减与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] += self.lazy[i] * (t - m)

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] += val * (t - s + 1)
            self.lazy[i] += val
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreeRangeAddMulSum:

    def __init__(self, p, n):
        self.p = p
        # 模板：区间值增减乘积与区间和查询
        self.cover = [0] * 4 * n
        self.lazy = [[] for _ in range(4 * n)]

    def push_down(self, i, s, m, t):

        if self.lazy[i]:
            for op, val in self.lazy[i]:
                if op == "add":
                    self.cover[2 * i] += val * (m - s + 1)
                    self.cover[2 * i + 1] += val * (t - m)

                    self.lazy[2 * i] += [[op, val]]
                    self.lazy[2 * i + 1] += [[op, val]]
                else:
                    self.cover[2 * i] *= val
                    self.cover[2 * i + 1] *= val

                    self.lazy[2 * i] += [[op, val]]
                    self.lazy[2 * i + 1] += [[op, val]]
                self.cover[2 * i] %= self.p
                self.cover[2 * i + 1] %= self.p

            self.lazy[i] = []

    def update(self, left, r, s, t, op, val, i):
        if left <= s and t <= r:
            if op == "add":
                self.cover[i] += val * (t - s + 1)
                self.lazy[i] += [["add", val]]
            else:
                self.cover[i] *= val
                self.lazy[i] += [["mul", val]]
            self.cover[i] %= self.p
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, op, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, op, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.cover[i] %= self.p
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreeRangeUpdateSum:
    def __init__(self):
        # 模板：区间值修改与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] = val * (t - s + 1)
            self.lazy[i] = val
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreePointAddSumMaxMin:
    def __init__(self, n: int):
        # 索引从 1 开始
        self.n = n
        self.min = [0] * (n * 4)
        self.sum = [0] * (n * 4)
        self.max = [0] * (n * 4)

    # 将 idx 上的元素值增加 val
    def add(self, o: int, l: int, r: int, idx: int, val: int) -> None:
        # 索引从 1 开始
        if l == r:
            self.min[o] += val
            self.sum[o] += val
            self.max[o] += val
            return
        m = (l + r) // 2
        if idx <= m:
            self.add(o * 2, l, m, idx, val)
        else:
            self.add(o * 2 + 1, m + 1, r, idx, val)
        self.min[o] = min(self.min[o * 2], self.min[o * 2 + 1])
        self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])
        self.sum[o] = self.sum[o * 2] + self.sum[o * 2 + 1]

    # 返回区间 [L,R] 内的元素和
    def query_sum(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.sum[o]
        sum_ = 0
        m = (l + r) // 2
        if L <= m:
            sum_ += self.query_sum(o * 2, l, m, L, R)
        if R > m:
            sum_ += self.query_sum(o * 2 + 1, m + 1, r, L, R)
        return sum_

    # 返回区间 [L,R] 内的最小值
    def query_min(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.min[o]
        res = 10 ** 9 + 7
        m = (l + r) // 2
        if L <= m:
            res = min(res, self.query_min(o * 2, l, m, L, R))
        if R > m:
            res = min(res, self.query_min(o * 2 + 1, m + 1, r, L, R))
        return res

    # 返回区间 [L,R] 内的最大值
    def query_max(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.max[o]
        res = 0
        m = (l + r) // 2
        if L <= m:
            res = max(res, self.query_max(o * 2, l, m, L, R))
        if R > m:
            res = max(res, self.query_max(o * 2 + 1, m + 1, r, L, R))
        return res


class SegmentTreeRangeChangeQueryOr:
    def __init__(self, n) -> None:
        # 模板：区间值与操作修改，区间值或查询
        self.n = n
        self.lazy = [0] * (4 * self.n)  # 懒标记与操作
        self.cover = [0] * (4 * self.n)  # 区间或操作初始值为 1
        return

    def make_tag(self, val: int, i: int) -> None:
        self.cover[i] = val
        self.lazy[i] = val
        return

    def push_down(self, i: int) -> None:
        # 下放懒标记
        if self.lazy[i]:
            self.make_tag(self.lazy[i], 2 * i)
            self.make_tag(self.lazy[i], 2 * i + 1)
            self.lazy[i] = 0

    def push_up(self, i: int) -> None:
        self.cover[i] = self.cover[2 * i] | self.cover[2 * i + 1]

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(val, i)
                    continue
                m = s + (t - s) // 2
                self.push_down(i)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def query_or(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans |= self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class SegmentTreeRangeUpdateMax:
    # 模板：持续修改区间值并求最大值
    def __init__(self):
        self.height = defaultdict(lambda: float("-inf"))
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.lazy[i]
            self.height[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = val
            self.lazy[i] = val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 * i] > self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l <= s and t <= r:
            return self.height[i]
        self.push_down(i)
        m = s + (t - s) // 2
        highest = float("-inf")
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur > highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur > highest:
                highest = cur
        return highest


class SegmentTreeRangeUpdateMulQuerySum:
    def __init__(self, nums: List[int], p) -> None:
        # 模板：区间值增减、区间值修改、区间最大值查询
        self.inf = float("inf")
        self.p = p
        self.n = len(nums)
        self.nums = nums
        self.lazy_add = [0] * (4 * self.n)  # 懒标记
        self.lazy_mul = [1] * (4 * self.n)  # 懒标记
        self.cover = [0] * (4 * self.n)  # 最大值
        self.build()  # 初始化线段树
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if s == t:
                    self.cover[i] = self.nums[s]
                else:
                    stack.append([s, t, i, 0])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * i, 1])
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def make_tag(self, s: int, t: int, x: int, flag: int, i: int) -> None:
        if flag == 1:  # 乘法
            self.lazy_mul[i] = (self.lazy_mul[i]*x) % self.p
            self.lazy_add[i] = (self.lazy_add[i]*x) % self.p
            self.cover[i] = (self.cover[i]*x) % self.p
        else:
            self.lazy_add[i] = (self.lazy_add[i]+x) % self.p
            self.cover[i] = (self.cover[i] + x*(t-s+1)) % self.p
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy_mul[i] != 1:
            self.make_tag(s, m, self.lazy_mul[i], 1, 2*i)
            self.make_tag(m+1, t, self.lazy_mul[i], 1, 2 * i+1)
            self.lazy_mul[i] = 1

        if self.lazy_add[i] != 0:
            self.make_tag(s, m, self.lazy_add[i], 2, 2 * i)
            self.make_tag(m + 1, t, self.lazy_add[i], 2, 2 * i + 1)
            self.lazy_add[i] = 0

    def update(self, left: int, right: int, s: int, t: int, val: int, flag: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    if flag == 1:
                        self.cover[i] = (self.cover[i] * val) % self.p
                        self.lazy_add[i] = (self.lazy_add[i] * val) % self.p
                        self.lazy_mul[i] = (self.lazy_mul[i] * val) % self.p
                    else:
                        self.cover[i] = (self.cover[i] + val * (t-s+1)) % self.p
                        self.lazy_add[i] = (self.lazy_add[i] + val) % self.p
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
                self.cover[i] %= self.p
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans % self.p


class SegmentTreeRangeSubConSum:
    def __init__(self, nums: List[int]) -> None:
        # 模板：单点修改、区间最大连续子段和查询
        self.inf = float("inf")
        self.n = len(nums)
        self.nums = nums
        self.cover = [-inf] * (4 * self.n)
        self.left = [-inf] * (4 * self.n)
        self.right = [-inf] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        self.build()  # 初始化线段树
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def check(self, a, b):
        if a >= 0 and b >= 0:
            return a+b
        return self.max(a, b)

    def push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.max(self.cover[2 * i], self.cover[2 * i + 1])
        self.cover[i] = self.max(self.cover[i], self.right[2 * i] + self.left[2 * i + 1])
        self.left[i] = self.max(self.left[2 * i], self.sum[2 * i] + self.left[2 * i + 1])
        self.right[i] = self.max(self.right[2 * i + 1], self.sum[2 * i+1] + self.right[2 * i])
        self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]
        return

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if s == t:
                    self.cover[i] = self.nums[s]
                    self.left[i] = self.nums[s]
                    self.right[i] = self.nums[s]
                    self.sum[i] = self.nums[s]
                    continue
                stack.append([s, t, i, 0])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i, 1])
                stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.cover[i] = val
                    self.left[i] = val
                    self.right[i] = val
                    self.sum[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append([s, t, i, 0])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间的最大连续和，注意这里是递归
        if left <= s and t <= right:
            return self.cover[i], self.left[i], self.right[i], self.sum[i]

        m = s + (t - s) // 2

        if right <= m:
            return self.query_max(left, right, s, m, 2 * i)
        if left > m:
            return self.query_max(left, right, m + 1, t, 2 * i + 1)

        res1 = self.query_max(left, right, s, m, 2 * i)
        res2 = self.query_max(left, right, m + 1, t, 2 * i + 1)

        # 参照区间递归方式
        res = [0]*4
        res[0] = self.max(res1[0], res2[0])
        res[0] = self.max(res[0], res1[2]+res2[1])
        res[1] = self.max(res1[1], res1[3]+res2[1])
        res[2] = self.max(res2[2], res2[3]+res1[2])
        res[3] = res1[3] + res2[3]
        return res


class SegmentTreeRangeUpdateSubConSum:
    def __init__(self, nums: List[int]) -> None:
        # 模板：区间修改、区间最大连续子段和查询
        self.inf = float("inf")
        self.n = len(nums)
        self.nums = nums
        self.cover = [-inf] * (4 * self.n)
        self.left = [-inf] * (4 * self.n)
        self.right = [-inf] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        self.lazy = [inf] * (4 * self.n)
        self.build()  # 初始化线段树
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def check(self, a, b):
        if a >= 0 and b >= 0:
            return a + b
        return self.max(a, b)

    def push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.max(self.cover[2 * i], self.cover[2 * i + 1])
        self.cover[i] = self.max(self.cover[i], self.right[2 * i] + self.left[2 * i + 1])
        self.left[i] = self.max(self.left[2 * i], self.sum[2 * i] + self.left[2 * i + 1])
        self.right[i] = self.max(self.right[2 * i + 1], self.sum[2 * i + 1] + self.right[2 * i])
        self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]
        return

    def make_tag(self, s, t, i, val):
        self.sum[i] = val * (t - s + 1)
        self.cover[i] = val if val < 0 else val * (t - s + 1)
        self.left[i] = val if val < 0 else val * (t - s + 1)
        self.right[i] = val if val < 0 else val * (t - s + 1)
        self.lazy[i] = val
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i] != self.inf:
            self.make_tag(s, m, 2 * i, self.lazy[i])
            self.make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = self.inf

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if s == t:
                    self.cover[i] = self.nums[s]
                    self.left[i] = self.nums[s]
                    self.right[i] = self.nums[s]
                    self.sum[i] = self.nums[s]
                    continue
                stack.append([s, t, i, 0])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i, 1])
                stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.cover[i] = val if val < 0 else val*(t-s+1)
                    self.left[i] = val if val < 0 else val*(t-s+1)
                    self.right[i] = val if val < 0 else val*(t-s+1)
                    self.sum[i] = val*(t-s+1)
                    self.lazy[i] = val
                    continue
                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间的最大连续和，注意这里是递归
        if left <= s and t <= right:
            return self.cover[i], self.left[i], self.right[i], self.sum[i]

        m = s + (t - s) // 2
        self.push_down(i, s, m, t)

        if right <= m:
            return self.query_max(left, right, s, m, 2 * i)
        if left > m:
            return self.query_max(left, right, m + 1, t, 2 * i + 1)

        res1 = self.query_max(left, right, s, m, 2 * i)
        res2 = self.query_max(left, right, m + 1, t, 2 * i + 1)

        # 参照区间递归方式
        res = [0] * 4
        res[0] = self.max(res1[0], res2[0])
        res[0] = self.max(res[0], res1[2] + res2[1])
        res[1] = self.max(res1[1], res1[3] + res2[1])
        res[2] = self.max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res


class SegmentTreeRangeUpdateMin:
    # 模板：持续修改区间值并求最小值
    def __init__(self):
        self.height = defaultdict(lambda: float("inf"))
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.lazy[i]
            self.height[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = val
            self.lazy[i] = val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 *i] < self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l <= s and t <= r:
            return self.height[i]
        self.push_down(i)
        m = s + (t - s) // 2
        highest = float("inf")
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur < highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur < highest:
                highest = cur
        return highest


class SegmentTreeRangeUpdateAvgDev:
    def __init__(self, n) -> None:
        # 模板：区间增减、区间平均值与区间方差
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.cover_2 = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.cover_2[i] = self.cover_2[2 * i] + self.cover_2[2 * i + 1]
        return

    def make_tag(self, s, t, i, val):
        self.cover_2[i] += self.cover[i]*2*val + (t-s+1)*val*val
        self.cover[i] += val*(t-s+1)
        self.lazy[i] += val
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.make_tag(s, m, 2 * i, self.lazy[i])
            self.make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = 0

    def build(self, nums) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if s == t:
                    self.cover[i] = nums[s]
                    self.cover_2[i] = nums[s]*nums[s]
                    continue
                stack.append([s, t, i, 0])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i, 1])
                stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增加 val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if left <= s and t <= right:
                    self.make_tag(s, t, i, val)
                    continue
                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, i, 0])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.push_up(i)
        return

    def query(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans1 = ans2 = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans1 += self.cover[i]
                ans2 += self.cover_2[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return [ans1, ans2]


class SegmentTreePointChangeLongCon:
    def __init__(self, n) -> None:
        # 模板：单点修改，查找最长的01交替字符子串连续区间
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.left_0 = [0] * (4 * self.n)
        self.left_1 = [0] * (4 * self.n)
        self.right_0 = [0] * (4 * self.n)
        self.right_1 = [0] * (4 * self.n)
        self.build()
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def push_up(self, i, s, m, t):
        # 合并区间的函数
        self.cover[i] = self.max(self.cover[2 * i], self.cover[2 * i + 1])
        self.cover[i] = self.max(self.cover[i], self.right_0[2 * i] + self.left_1[2 * i + 1])
        self.cover[i] = self.max(self.cover[i], self.right_1[2 * i] + self.left_0[2 * i + 1])

        self.left_0[i] = self.left_0[2 * i]
        if self.left_0[2 * i] == m - s + 1:
            self.left_0[i] += self.left_0[2 * i + 1] if (m - s + 1) % 2 == 0 else self.left_1[2 * i + 1]

        self.left_1[i] = self.left_1[2 * i]
        if self.left_1[2 * i] == m - s + 1:
            self.left_1[i] += self.left_1[2 * i + 1] if (m - s + 1) % 2 == 0 else self.left_0[2 * i + 1]

        self.right_0[i] = self.right_0[2 * i + 1]
        if self.right_0[2 * i + 1] == t - m:
            self.right_0[i] += self.right_0[2 * i] if (t - m) % 2 == 0 else self.right_1[2 * i]

        self.right_1[i] = self.right_1[2 * i + 1]
        if self.right_1[2 * i + 1] == t - m:
            self.right_1[i] += self.right_1[2 * i] if (t - m) % 2 == 0 else self.right_0[2 * i]
        return

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = 1
                    self.left_0[i] = 1
                    self.right_0[i] = 1
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                m = s + (t - s) // 2
                self.push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增加 val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.left_0[i] = 1 - self.left_0[i]
                    self.right_0[i] = 1 - self.right_0[i]
                    self.left_1[i] = 1 - self.left_1[i]
                    self.right_1[i] = 1 - self.right_1[i]
                    self.cover[i] = 1
                    continue
                m = s + (t - s) // 2
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                m = s + (t - s) // 2
                self.push_up(i, s, m, t)
        return

    def query(self):
        return self.cover[1]


class SegmentTreeRangeAndOrXOR:
    def __init__(self, n) -> None:
        # 模板：区间修改成01或者反转，区间查询最多有多少连续的1，以及总共有多少1
        self.n = n
        self.cover_1 = [0] * (4 * self.n)
        self.cover_0 = [0] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        self.left_1 = [0] * (4 * self.n)
        self.right_1 = [0] * (4 * self.n)
        self.left_0 = [0] * (4 * self.n)
        self.right_0 = [0] * (4 * self.n)
        self.lazy = [inf] * (4 * self.n)
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def push_up(self, i, s, m, t):
        # 合并区间的函数
        self.cover_1[i] = self.max(self.cover_1[2 * i], self.cover_1[2 * i + 1])
        self.cover_1[i] = self.max(self.cover_1[i], self.right_1[2*i]+self.left_1[2*i+1])

        self.cover_0[i] = self.max(self.cover_0[2 * i], self.cover_0[2 * i + 1])
        self.cover_0[i] = self.max(self.cover_0[i], self.right_0[2 * i] + self.left_0[2 * i + 1])

        self.sum[i] = self.sum[2*i] + self.sum[2*i+1]

        self.left_1[i] = self.left_1[2*i]
        if self.left_1[i] == m-s+1:
            self.left_1[i] += self.left_1[2*i+1]

        self.left_0[i] = self.left_0[2 * i]
        if self.left_0[i] == m - s + 1:
            self.left_0[i] += self.left_0[2 * i + 1]

        self.right_1[i] = self.right_1[2 * i+1]
        if self.right_1[i] == t-m:
            self.right_1[i] += self.right_1[2 * i]

        self.right_0[i] = self.right_0[2 * i+1]
        if self.right_0[i] == t-m:
            self.right_0[i] += self.right_0[2 * i]

        return

    def make_tag(self, s, t, i, val):
        if val == 0:
            self.cover_1[i] = 0
            self.sum[i] = 0
            self.cover_0[i] = t-s+1
            self.left_1[i] = self.right_1[i] = 0
            self.right_0[i] = self.left_0[i] = t-s+1
        elif val == 1:
            self.cover_1[i] = t-s+1
            self.cover_0[i] = 0
            self.sum[i] = t-s+1
            self.left_1[i] = self.right_1[i] = t-s+1
            self.right_0[i] = self.left_0[i] = 0
        else:
            self.cover_1[i], self.cover_0[i] = self.cover_0[i], self.cover_1[i]
            self.sum[i] = t-s+1-self.sum[i]
            self.left_0[i], self.left_1[i] = self.left_1[i], self.left_0[i]
            self.right_0[i], self.right_1[i] = self.right_1[i], self.right_0[i]
        self.lazy[i] = val
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i] != inf:
            self.make_tag(s, m, 2 * i, self.lazy[i])
            self.make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = inf

    def build(self, nums) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    if nums[s] == 0:
                        self.cover_1[i] = 0
                        self.sum[i] = 0
                        self.cover_0[i] = 1
                        self.left_1[i] = self.right_1[i] = 0
                        self.left_0[i] = self.right_0[i] = 1
                    else:
                        self.cover_1[i] = 1
                        self.sum[i] = 1
                        self.cover_0[i] = 0
                        self.left_1[i] = self.right_1[i] = 1
                        self.left_0[i] = self.right_0[i] = 0
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                m = s + (t - s) // 2
                self.push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增加 val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                m = s + (t - s) // 2
                if s < t:
                    self.push_down(i, s, m, t)
                if left <= s and t <= right:
                    self.make_tag(s, t, i, val)
                    continue

                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                m = s + (t - s) // 2
                self.push_up(i, s, m, t)
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans

    def query_max_length(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间的最大连续和，注意这里是递归
        if left <= s and t <= right:
            return self.cover_1[i], self.left_1[i], self.right_1[i]

        m = s + (t - s) // 2
        self.push_down(i, s, m, t)

        if right <= m:
            return self.query_max_length(left, right, s, m, 2 * i)
        if left > m:
            return self.query_max_length(left, right, m + 1, t, 2 * i + 1)

        res1 = self.query_max_length(left, right, s, m, 2 * i)
        res2 = self.query_max_length(left, right, m + 1, t, 2 * i + 1)

        # 参照区间递归方式
        res = [0] * 3
        res[0] = self.max(res1[0], res2[0])
        res[0] = self.max(res[0], res1[2] + res2[1])

        res[1] = res1[1]
        if res[1] == m-s+1:
            res[1] += res2[1]

        res[2] = res2[2]
        if res[2] == t-m:
            res[2] += res1[2]
        return res


class SegmentTreeRangeSqrtSum:
    def __init__(self, n):
        # 模板：区间值开方向下取整，区间和查询
        self.inf = float("inf")
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [self.inf] * (4 * self.n)  # 懒标记

    def build(self, nums):
        stack = [[0, self.n - 1, 1, 1]]
        while stack:
            s, t, ind, state = stack.pop()
            if state:
                if s == t:
                    self.cover[ind] = nums[s]
                else:
                    stack.append([s, t, ind, 0])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind, 1])
                    stack.append([m + 1, t, 2 * ind + 1, 1])
            else:
                self.cover[ind] = self.cover[2 * ind] + self.cover[2 * ind + 1]
        return

    def change(self, left, right, s, t, i):
        # 更新区间值
        stack = [[s, t, i, 1]]
        while stack:
            s, t, i, state = stack.pop()
            if state:
                if self.cover[i] == t-s+1:
                    continue
                if s == t:
                    self.cover[i] = int(self.cover[i]**0.5)
                    continue
                stack.append([s, t, i, 0])
                m = s + (t - s) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i, 1])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1, 1])
            else:
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query_sum(self, left, right, s, t, i):
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6358(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # 模板：区间进行 0 1 翻转与 1 的个数查询
        res = []
        seg = SegBitSet()
        n = len(nums1)
        for i in range(n):
            if nums1[i]:
                seg.update(i, i)
        s = sum(nums2)
        for a, b, c in queries:
            if a == 1:
                seg.update(b, c)
            elif a == 2:
                s += seg.val.bit_count() * b
            else:
                res.append(s)
        return res

    @staticmethod
    def lg_p1904(ac=FastIO()):

        # 模板：使用线段树，区间更新最大值并单点查询计算天际线
        low = 0
        high = 10 ** 4
        segment = SegmentTreeRangeAddMax(high)
        nums = set()
        while True:
            s = ac.read_str()
            if not s:
                break
            x, h, y = [int(w) for w in s.split() if w]
            nums.add(x)
            nums.add(y)
            segment.update(x, y - 1, low, high, h, 1)
        nums = sorted(list(nums))
        n = len(nums)
        height = [segment.query(num, num, low, high, 1) for num in nums]
        ans = []
        pre = -1
        for i in range(n):
            if height[i] != pre:
                ans.extend([nums[i], height[i]])
                pre = height[i]
        ac.lst(ans)
        return

    @staticmethod
    def lc_218(buildings: List[List[int]]) -> List[List[int]]:
        # 模板：线段树离散化区间且持续增加最大值
        pos = set()
        for left, right, _ in buildings:
            pos.add(left)
            pos.add(right)
        lst = sorted(list(pos))
        n = len(lst)
        dct = {x: i for i, x in enumerate(lst)}
        # 离散化更新线段树
        segment = SegmentTreeRangeAddMax(n)
        for left, right, height in buildings:
            segment.update(dct[left], dct[right]-1, 0, n-1, height, 1)
        # 按照端点进行关键点查询
        pre = -1
        ans = []
        for pos in lst:
            h = segment.query(dct[pos], dct[pos], 0, n-1, 1)
            if h != pre:
                ans.append([pos, h])
                pre = h
        return ans

    @staticmethod
    def cf_380c(ac=FastIO()):
        word = []
        queries = []
        # 模板：线段树进行分治并使用dp合并
        n = len(word)
        a = [0] * (4 * n)
        b = [0] * (4 * n)
        c = [0] * (4 * n)

        @ac.bootstrap
        def update(left, r, s, t, i):
            if s == t:
                if word[s - 1] == ")":
                    c[i] = 1
                else:
                    b[i] = 1
                a[i] = 0
                yield

            m = s + (t - s) // 2
            if left <= m:
                yield update(left, r, s, m, i << 1)
            if r > m:
                yield update(left, r, m + 1, t, i << 1 | 1)

            match = min(b[i << 1], c[i << 1 | 1])
            a[i] = a[i << 1] + a[i << 1 | 1] + 2 * match
            b[i] = b[i << 1] + b[i << 1 | 1] - match
            c[i] = c[i << 1] + c[i << 1 | 1] - match
            yield

        @ac.bootstrap
        def query(left, r, s, t, i):
            if left <= s and t <= r:
                d[i] = [a[i], b[i], c[i]]
                yield

            a1 = b1 = c1 = 0
            m = s + (t - s) // 2
            if left <= m:
                yield query(left, r, s, m, i << 1)
                a2, b2, c2 = d[i << 1]
                match = min(b1, c2)
                a1 += a2 + 2 * match
                b1 += b2 - match
                c1 += c2 - match
            if r > m:
                yield query(left, r, m + 1, t, i << 1 | 1)
                a2, b2, c2 = d[i << 1 | 1]
                match = min(b1, c2)
                a1 += a2 + 2 * match
                b1 += b2 - match
                c1 += c2 - match
            d[i] = [a1, b1, c1]
            yield

        update(1, n, 1, n, 1)
        ans = []
        for x, y in queries:
            d = defaultdict(list)
            query(x, y, 1, n, 1)
            ans.append(d[1][0])
        return ans

    @staticmethod
    def lg_p3372(ac=FastIO()):
        # 模板：线段树 区间增减 与区间和查询
        n, m = ac.read_ints()
        segment = SegmentTreeRangeUpdateQuerySumMinMax(ac.read_list_ints())

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                segment.update(x-1, y-1, 0, n-1, k, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_sum(x-1, y-1, 0, n-1, 1))
        return

    @staticmethod
    def lg_p3870(ac=FastIO()):
        # 模板：区间异或 0 与 1 翻转
        n, m = ac.read_ints()
        segment = SegmentTreeRangeUpdateXORSum(n)

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                x, y = lst[1:]
                segment.update(x-1, y-1, 0, n-1, 1, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_sum(x-1, y-1, 0, n-1, 1))
        return

    @staticmethod
    def lg_p1438(ac=FastIO()):
        # 模板：差分数组区间增减加线段树查询区间和
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        segment = SegmentTreeRangeUpdateQuerySumMinMax([0] * n)

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k, d = lst[1:]
                if x == y:
                    segment.update(x - 1, x - 1, 0, n - 1, k, 1)
                    if y <= n - 1:
                        segment.update(y, y, 0, n - 1, -k, 1)
                else:
                    segment.update(x - 1, x - 1, 0, n - 1, k, 1)
                    segment.update(x, y - 1, 0, n - 1, d, 1)
                    cnt = y-x
                    if y <= n - 1:
                        segment.update(y, y, 0, n - 1, -cnt*d-k, 1)
            else:
                x = lst[1]
                ac.st(segment.query_sum(0, x - 1, 0, n - 1, 1) + nums[x - 1])
        return

    @staticmethod
    def lg_p1253(ac=FastIO()):

        # 模板：区间增减与区间修改并使用线段树查询区间和
        n, m = ac.read_ints()
        segment = SegmentTreeRangeUpdateChangeQueryMax(ac.read_list_ints())

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                segment.update(x - 1, y - 1, 0, n - 1, k, 1, 1)
            elif lst[0] == 2:
                x, y, k = lst[1:]
                segment.update(x - 1, y - 1, 0, n - 1, k, 2, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_max(x - 1, y - 1, 0, n - 1, 1))
        return

    @staticmethod
    def lg_p3373(ac=FastIO()):

        # 模板：区间乘法与区间加法并使用线段树查询区间和
        n, m, p = ac.read_ints()
        nums = ac.read_list_ints()
        segment = SegmentTreeRangeUpdateMulQuerySum(nums, p)
        stack = deque()
        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 3:
                while stack:
                    op, x, y, k = stack.popleft()
                    segment.update(x - 1, y - 1, 0, n - 1, k, op, 1)
                x, y = lst[1:]
                ac.st(segment.query_sum(x-1, y-1, 0, n-1, 1))
            else:
                stack.append(lst)
        return

    @staticmethod
    def lg_p4513(ac=FastIO()):

        # 模板：单点修改后区间查询最大的子段和
        n, m = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]
        segment = SegmentTreeRangeSubConSum(nums)
        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                a, b = lst[1:]
                a, b = ac.min(a, b), ac.max(a, b)
                ans = segment.query_max(a-1, b-1, 0, n-1, 1)[0]
                ac.st(ans)
            else:
                a, s = lst[1:]
                segment.update(a-1, a-1, 0, n-1, s, 1)
                nums[a-1] = s
        return

    @staticmethod
    def lg_p1471(ac=FastIO()):
        # 模板：区间增减，维护区间和与区间数字平方的和，以计算均差与方差
        n, m = ac.read_ints()
        tree = SegmentTreeRangeUpdateAvgDev(n)
        tree.build(ac.read_list_floats())
        for _ in range(m):
            lst = ac.read_list_floats()
            if lst[0] == 1:
                x, y, k = lst[1:]
                x = int(x)
                y = int(y)
                tree.update(x-1, y-1, 0, n-1, k, 1)
            elif lst[0] == 2:
                x, y = lst[1:]
                x = int(x)
                y = int(y)
                ans = (tree.query(x-1, y-1, 0, n-1, 1)[0])/(y-x+1)
                ac.st("%.4f" % ans)
            else:
                x, y = lst[1:]
                x = int(x)
                y = int(y)
                avg, avg_2 = tree.query(x - 1, y - 1, 0, n - 1, 1)
                ans = -(avg*avg/(y-x+1))**2 + avg_2/(y-x+1)
                ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p6492(ac=FastIO()):
        # 模板：单点修改，查找最长的01交替字符子串连续区间
        n, q = ac.read_ints()
        tree = SegmentTreePointChangeLongCon(n)
        for _ in range(q):
            i = ac.read_int() - 1
            tree.update(i, i, 0, n-1, 1)
            ac.st(tree.query())
        return

    @staticmethod
    def lg_p4145(ac=FastIO()):
        # 模板：区间值开方向下取整，区间和查询
        n = ac.read_int()
        tree = SegmentTreeRangeSqrtSum(n)
        tree.build(ac.read_list_ints())
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            a, b = [int(w) - 1 for w in lst[1:]]
            if a > b:
                a, b = b, a
            if lst[0] == 0:
                tree.change(a, b, 0, n-1, 1)
            else:
                ac.st(tree.query_sum(a, b, 0, n-1, 1))
        return

    @staticmethod
    def lg_2572(ac=FastIO()):
        # 模板：区间修改成01或者反转，区间查询最多有多少连续的1，以及总共有多少1
        def check(tmp):
            ans = pre = 0
            for num in tmp:
                if num:
                    pre += 1
                else:
                    ans = ans if ans > pre else pre
                    pre = 0
            ans = ans if ans > pre else pre
            return ans

        # n, m = ac.read_ints()
        for s in range(100):
            # random.seed(s)
            print(f"seed_{s}")
            n = random.randint(1, 1000)
            m = random.randint(1, 1000)
            tree = SegmentTreeRangeAndOrXOR(n)
            # tree.build(ac.read_list_ints())
            nums = [random.randint(1, 1) for _ in range(n)]
            tree.build(nums)
            for _ in range(m):
                lst = [random.randint(0, 4), random.randint(0, n-1), random.randint(0, n-1)]
                # lst = ac.read_list_ints()
                left, right = lst[1:]
                if left > right:
                    left, right = right, left
                if lst[0] <= 2:
                    tree.update(left, right, 0, n-1, lst[0], 1)
                    if lst[0] == 0:
                        for i in range(left, right+1):
                            nums[i] = 0
                    elif lst[0] == 1:
                        for i in range(left, right+1):
                            nums[i] = 1
                    else:
                        for i in range(left, right+1):
                            nums[i] = 1-nums[i]
                    assert nums == [tree.query_sum(i, i, 0, n-1, 1) for i in range(n)]
                elif lst[0] == 3:
                    #print("\n")
                    #print(nums)
                    #print([tree.query_sum(i, i, 0, n-1, 1) for i in range(n)])
                    assert sum(nums[left:right+1]) == tree.query_sum(left, right, 0, n-1, 1)
                    # ac.st(tree.query_sum(left, right, 0, n-1, 1))
                else:
                    #print("\n")
                    #print(nums)
                    #print([tree.query_sum(i, i, 0, n-1, 1) for i in range(n)])
                    #print(tree.query_max_length(left, right, 0, n - 1, 1)[0], check(nums[left:right+1]))
                    assert tree.query_max_length(left, right, 0, n - 1, 1)[0] == check(nums[left:right+1])
                    # ac.st(tree.query_max_length(left, right, 0, n - 1, 1)[0])
        return

    @staticmethod
    def lg_p1558(ac=FastIO()):
        # 模板：线段树区间值修改，区间或值查询
        n, t, q = ac.read_ints()
        tree = SegmentTreeRangeChangeQueryOr(n)
        tree.update(0, n-1, 0, n-1, 1, 1)
        for _ in range(q):
            lst = ac.read_list_strs()
            if lst[0] == "C":
                a, b, c = [int(w) for w in lst[1:]]
                if a > b:
                    a, b = b, a
                tree.update(a - 1, b - 1, 0, n - 1, 1 << (c - 1), 1)
            else:
                a, b = [int(w) for w in lst[1:]]
                if a > b:
                    a, b = b, a
                ac.st(bin(tree.query_or(a - 1, b - 1, 0, n - 1, 1)).count("1"))
        return


class TestGeneral(unittest.TestCase):

    def test_segment_tree_range_add_sum(self):
        low = 0
        high = 10**9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeAddSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_sum(self):
        low = 0
        high = 10**9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeUpdateSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_point_add_sum_max_min(self):
        low = 0
        high = 10000

        nums = [random.randint(low, high) for _ in range(high)]
        staasmm = SegmentTreePointAddSumMaxMin(high)
        for i in range(high):
            staasmm.add(1, low, high, i + 1, nums[i])

        for _ in range(high):
            # 单点进行增减值
            i = random.randint(0, high - 1)
            num = random.randint(low, high)
            nums[i] += num
            staasmm.add(1, low, high, i + 1, num)

            # 查询区间和、最大值、最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert staasmm.query_sum(
                1, low, high, left + 1, right + 1) == sum(nums[left:right + 1])
            assert staasmm.query_max(
                1, low, high, left + 1, right + 1) == max(nums[left:right + 1])
            assert staasmm.query_min(
                1, low, high, left + 1, right + 1) == min(nums[left:right + 1])

            # 查询单点和、最大值、最小值
            left = random.randint(0, high - 1)
            right = left
            assert staasmm.query_sum(
                1, low, high, left + 1, right + 1) == sum(nums[left:right + 1])
            assert staasmm.query_max(
                1, low, high, left + 1, right + 1) == max(nums[left:right + 1])
            assert staasmm.query_min(
                1, low, high, left + 1, right + 1) == min(nums[left:right + 1])

        return

    def test_segment_tree_range_add_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeAddMax(high)
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMax()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMin()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])
        return

    def test_segment_tree_range_add_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeAddMin(high)
        for i in range(high):
            stra.update(i, i, low, high-1, nums[i], 1)
            assert stra.query(i, i, low, high-1, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high-1, 1) == min(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert stra.query(left, right, low, high-1, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
        return

    def test_segment_tree_range_change_query_sum_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateQuerySumMinMax(nums)
        for i in range(high):
            assert stra.query_min(i, i, low, high-1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high-1, 1) == sum(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high-1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high-1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high-1, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateQuerySumMinMax(nums)

        assert stra.query_min(0, high - 1, low, high - 1, 1) == min(nums)
        assert stra.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert stra.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert stra.query_min(i, i, low, high-1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high-1, 1) == sum(nums[i:i + 1])
            assert stra.query_max(i, i, low, high-1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high -1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.update(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_change_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeChangeQuerySumMinMax(nums)

        assert stra.query_min(0, high-1, low, high - 1, 1) == min(nums)
        assert stra.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert stra.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert stra.query_min(i, i, low, high-1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high-1, 1) == sum(nums[i:i + 1])
            assert stra.query_max(i, i, low, high-1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.change(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high -1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.change(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_sub_con_max(self):

        def check(lst):
            pre = ans = lst[0]
            for x in lst[1:]:
                pre = pre + x if pre + x > x else x
                ans = ans if ans > pre else pre
            return ans

        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateSubConSum(nums)

        assert stra.query_max(0, high - 1, low, high - 1, 1)[0] == check(nums)

        for _ in range(high):
            # 区间更新值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.update(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_max(left, right, low, high - 1, 1)[0] == check(nums[left:right + 1])
        return


if __name__ == '__main__':
    unittest.main()
