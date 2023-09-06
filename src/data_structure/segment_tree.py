import random
import unittest
from collections import defaultdict, deque
from typing import List

from sortedcontainers import SortedList

from src.basis.binary_search import BinarySearch
from src.fast_io import inf, FastIO

"""
算法：线段树
功能：用以修改和查询区间的值信息，支持增减、修改，区间和、区间最大值、区间最小值、动态开点线段树（即使用defaultdict而不是数组实现）
题目：

===================================力扣===================================
218. 天际线问题（https://leetcode.cn/problems/the-skyline-problem/solution/by-liupengsay-isfo/）区间值修改与计算最大值
2286. 以组为单位订音乐会的门票（https://leetcode.cn/problems/booking-concert-tickets-in-groups/）区间值增减与计算区间和、区间最大值、区间最小值
2407. 最长递增子序列 II（https://leetcode.cn/problems/longest-increasing-subsequence-ii/）维护与查询区间最大值，然后进行DP更新
2179. 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2158. 每天绘制新区域的数量（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）线段树维护区间范围的覆盖
6318. 完成所有任务的最少时间（https://leetcode.cn/contest/weekly-contest-336/problems/minimum-time-to-complete-all-tasks/）线段树，贪心加二分
732. 我的日程安排表 III（https://leetcode.cn/problems/my-calendar-iii/）使用defaultdict进行动态开点线段树
1851. 包含每个查询的最小区间（https://leetcode.cn/problems/minimum-interval-to-include-each-query/）区间更新最小值、单点查询，也可以用离线查询与优先队列维护计算
2213. 由单个字符重复的最长子字符串（https://leetcode.cn/problems/longest-substring-of-one-repeating-character/）单点字母更新，最长具有相同字母的连续子数组查询
2276. 统计区间中的整数数目（https://leetcode.cn/problems/count-integers-in-intervals/）动态开点线段树模板题，维护区间并集的长度，也可使用SortedList
1340. 跳跃游戏 V（https://leetcode.cn/problems/jump-game-v/）可以使用线段树DP进行解决
2569. 更新数组后处理求和查询（https://leetcode.cn/problems/handling-sum-queries-after-update/）经典01线段树区间翻转与求和，也可以使用BitSet


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
P3740 [HAOI2014]贴海报（https://www.luogu.com.cn/problem/P3740）离散化线段树区间修改与单点查询
P4588 [TJOI2018]数学计算（https://www.luogu.com.cn/problem/P4588）转化为线段树单点值修改与区间乘积取模
P6627 [省选联考 2020 B 卷] 幸运数字（https://www.luogu.com.cn/problem/P6627）线段树维护和查询区间异或值
P8081 [COCI2011-2012#4] ZIMA（https://www.luogu.com.cn/problem/P8081）差分计数计算作用域，也可以线段树区间修改、区间加和查询
P8812 [蓝桥杯 2022 国 C] 打折（https://www.luogu.com.cn/problem/P8812）线段树查询和更新区间最小值
P8856 [POI2002]火车线路（https://www.luogu.com.cn/problem/solution/P8856）区间增减与区间最大值查询

================================CodeForces================================

https://codeforces.com/problemset/problem/482/B（区间按位或赋值、按位与查询）
C. Sereja and Brackets（https://codeforces.com/problemset/problem/380/C）线段树查询区间内所有合法连续子序列括号串的总长度
C. Circular RMQ（https://codeforces.com/problemset/problem/52/C）线段树更新和查询循环数组区间最小值
D. The Child and Sequence（https://codeforces.com/problemset/problem/438/D）使用线段树维护区间取模，区间和，修改单点值，和区间最大值
E. A Simple Task（https://codeforces.com/contest/558/problem/E）26个线段树维护区间排序信息
D. Water Tree（https://codeforces.com/problemset/problem/343/D）dfs序加线段树
E. XOR on Segment（https://codeforces.com/problemset/problem/242/E）线段树区间异或，与区间加和
C. Three displays（https://codeforces.com/problemset/problem/987/C）枚举中间数组，使用线段树维护前后缀最小值

================================AcWing================================
3805. 环形数组（https://www.acwing.com/problem/content/3808/）区间增减与最小值查询
5037. 区间异或（https://www.acwing.com/problem/content/5040/）同CF242E，使用二十多个01线段树维护区间异或与区间加和


参考：OI WiKi（xx）
"""


class SegmentTreeBitSet:
    # 使用位运算模拟线段树进行区间01翻转操作
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # 索引从0开始翻转区间[b, c]
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        # 索引从0开始查询区间[b, c]的个数
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
        stack = [[s, t, i]]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self.height[i] = self.max(self.height[i], val)
                    self.lazy[i] = self.max(self.lazy[i], val)
                    continue
                self.push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
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


class SegmentTreeUpdateQueryMin:
    # 模板：线段树区间更新、持续减小最小值
    def __init__(self, n):
        self.height = [inf]*(4*n)
        self.lazy = [inf]*(4*n)
        self.n = n

    def build(self, nums: List[int]):
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.make_tag(ind, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind)
        return

    def get(self):
        # 查询区间的所有值
        stack = [[0, self.n-1, 1]]
        nums = [inf]*self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.height[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i)
            stack.append([s, m, 2 * i])
            stack.append([m + 1, t, 2 * i + 1])
        return nums

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def push_down(self, i):
        # 懒标记下放，注意取最小值
        if self.lazy[i] != inf:
            self.height[2 * i] = self.min(self.height[2 * i], self.lazy[i])
            self.height[2 * i + 1] = self.min(self.height[2 * i + 1], self.lazy[i])

            self.lazy[2 * i] = self.min(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self.min(self.lazy[2 * i + 1], self.lazy[i])

            self.lazy[i] = inf
        return

    def make_tag(self, i, val):
        self.height[i] = self.min(self.height[i], val)
        self.lazy[i] = self.min(self.lazy[i], val)
        return

    def push_up(self, i):
        self.height[i] = self.min(self.height[2 * i], self.height[2 * i + 1])
        return

    def update_range(self, left, right, s, t, val, i):
        # 更新区间最小值
        stack = [[s, t, i]]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self.make_tag(i, val)
                    continue

                self.push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update_point(self, left, right, s, t, val, i):
        # 更新单点最小值
        while True:
            if left <= s and t <= right:
                self.make_tag(i, val)
                break
            self.push_down(i)
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1

        while i > 1:
            i //= 2
            self.push_up(i)
        return

    def query_range(self, left, right, s, t, i):
        # 查询区间的最小值
        stack = [[s, t, i]]
        floor = inf
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

    def query_point(self, left, right, s, t, i):
        # 查询单点的最小值
        a, b, i = s, t, i
        while True:
            if left <= a and b <= right:
                ans = self.height[i]
                break
            self.push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                a, b, i = a, m, 2 * i
            if right > m:
                a, b, i = m + 1, b, 2 * i + 1
        return ans


class SegmentTreeRangeUpdateQuerySumMinMax:
    def __init__(self, n) -> None:
        # 模板：区间值增减、区间和查询、区间最小值查询、区间最大值查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记只能初始化为0
        self.floor = [0] * (4 * self.n)  # 最小值也可初始化为inf
        self.ceil = [0] * (4 * self.n)  # 最大值也可初始化为-inf
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self, nums: List[int]) -> None:
        # 使用数组初始化线段树
        assert self.n == len(nums)
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.make_tag(ind, s, t, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind)
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

    def push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def make_tag(self, i, s, t, val) -> None:
        self.cover[i] += val * (t - s + 1)
        self.floor[i] += val
        self.ceil[i] += val
        self.lazy[i] += val
        return

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减单点值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始

        while True:
            if left <= s and t <= right:
                self.make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
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
        highest = inf
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
        highest = -inf
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

    def get_all_nums(self) -> List[int]:
        # 查询区间的所有值
        stack = [[0, self.n-1, 1]]
        nums = [0]*self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            stack.append([s, m, 2 * i])
            stack.append([m + 1, t, 2 * i + 1])
        return nums


class SegmentTreeRangeChangeQuerySumMinMax:
    def __init__(self, nums):
        # 模板：区间值修改、区间和查询、区间最小值查询、区间最大值查询
        self.n = len(nums)
        self.nums = nums
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [inf] * (4 * self.n)  # 懒标记只能初始化为inf
        self.floor = [0] * (4 * self.n)  # 最小值也可初始化为inf
        self.ceil = [0] * (4 * self.n)  # 最大值也可初始化为-inf
        self.build()  # 初始化数组

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def build(self):

        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.make_tag(ind, s, t, self.nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind)
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i] != inf:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.floor[2 * i] = self.lazy[i]
            self.floor[2 * i + 1] = self.lazy[i]

            self.ceil[2 * i] = self.lazy[i]
            self.ceil[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = inf

    def push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def make_tag(self, i, s, t, val) -> None:
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy[i] = val
        return

    def change(self, left, right, s, t, val, i):
        # 更新区间值
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def change_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减单点值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始

        while True:
            if left <= s and t <= right:
                self.make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
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
        highest = inf
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
        highest = -inf
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

    def get_all_nums(self):
        # 查询区间的所有值
        stack = [[0, self.n-1, 1]]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self.nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            stack.append([s, m, 2 * i])
            stack.append([m + 1, t, 2 * i + 1])
        return


class SegmentTreeRangeChangeQuerySumMinMaxDefaultDict:
    def __init__(self):
        # 模板：区间值修改、区间和查询、区间最小值查询、区间最大值查询（动态开点线段树）
        self.cover = defaultdict(int)  # 区间和  # 注意初始化值
        self.lazy = defaultdict(int)  # 懒标记  # 注意初始化值
        self.floor = defaultdict(int)  # 最小值  # 注意初始化值
        self.ceil = defaultdict(int)  # 最大值  # 注意初始化值

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.floor[2 * i] = self.lazy[i]
            self.floor[2 * i + 1] = self.lazy[i]

            self.ceil[2 * i] = self.lazy[i]
            self.ceil[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0

    def push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def make_tag(self, i, s, t, val) -> None:
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy[i] = val
        return

    def update_range(self, left, right, s, t, val, i):
        # 修改区间值
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改单点值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始

        while True:
            if left <= s and t <= right:
                self.make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
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
        highest = inf
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
        highest = -inf
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


class SegmentTreeRangeUpdateQuerySum:
    def __init__(self, n) -> None:
        # 模板：区间修改、区间和查询
        self.n = n
        self.sum = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    def push_up(self, i):
        # 合并区间的函数
        self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]
        return

    def make_tag(self, s, t, i, val):
        self.sum[i] = val * (t - s + 1)
        self.lazy[i] = val
        return

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.make_tag(s, m, 2 * i, self.lazy[i])
            self.make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = 0

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(s, t, i, val)
                    continue
                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def query_range(self, left: int, right: int, s: int, t: int, i: int):
        # 区间加和查询
        ans = 0
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class SegmentTreeRangeUpdateChangeQueryMax:
    def __init__(self, nums: List[int]) -> None:
        # 模板：区间值增减、区间值修改、区间最大值查询
        self.n = len(nums)
        self.nums = nums
        self.lazy = [[inf, 0]] * (4 * self.n)  # 懒标记
        self.ceil = [-inf] * (4 * self.n)  # 最大值
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
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.ceil[ind] = self.nums[s]
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.ceil[ind] = self.max(self.ceil[2 * ind], self.ceil[2 * ind + 1])
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i] != [inf, 0]:
            a, b = self.lazy[i]  # 分别表示修改为 a 与 增加 b
            if a == inf:
                self.ceil[2 * i] += b
                self.ceil[2 * i + 1] += b
                self.lazy[2 * i] = [inf, self.lazy[2 * i][1] + b]
                self.lazy[2 * i + 1] = [inf, self.lazy[2 * i + 1][1] + b]
            else:
                self.ceil[2 * i] = a
                self.ceil[2 * i + 1] = a
                self.lazy[2 * i] = [a, 0]
                self.lazy[2 * i + 1] = [a, 0]
            self.lazy[i] = [inf, 0]

    def update(self, left: int, right: int, s: int, t: int, val: int, flag: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    if flag == 1:
                        self.ceil[i] = val
                        self.lazy[i] = [val, 0]
                    elif self.lazy[i][0] != inf:
                        self.ceil[i] += val
                        self.lazy[i] = [self.lazy[i][0]+val, 0]
                    else:
                        self.ceil[i] += val
                        self.lazy[i] = [inf, self.lazy[i][1]+val]
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.ceil[i] = self.max(self.ceil[2 * i], self.ceil[2 * i + 1])
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int) -> int:

        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = -inf
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

    def build(self, nums) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                else:
                    stack.append([s, t, ~i])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * i])
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2 * i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意使用异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意使用异或抵消查询

            self.lazy[i] = 0
        return

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy[i] ^= val  # 注意使用异或抵消查询
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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
        # 修改区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
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
        # 模板：区间值增减、区间值乘法、区间值修改、区间最大值查询
        self.p = p
        self.n = len(nums)
        self.nums = nums
        self.lazy_add = [0] * (4 * self.n)  # 懒标记
        self.lazy_mul = [1] * (4 * self.n)  # 懒标记
        self.cover = [0] * (4 * self.n)  # 区间和
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
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = self.nums[s]
                else:
                    stack.append([s, t, ~i])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * i])
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
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
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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


class SegmentTreePointUpdateRangeMulQuery:
    def __init__(self, n, mod) -> None:
        # 模板：单点值修改、区间乘取模
        self.n = n
        self.mod = mod
        self.cover = [1] * (4 * self.n)  # 区间乘积取模
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改单点值 left == right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] * self.cover[2 * i + 1]
                self.cover[i] %= self.mod
        return

    def query_mul(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的乘积
        stack = [[s, t, i]]
        ans = 1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans *= self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class SegmentTreeRangeSubConSum:
    def __init__(self, nums: List[int]) -> None:
        # 模板：单点修改、区间最大连续子段和查询
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
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = self.nums[s]
                    self.left[i] = self.nums[s]
                    self.right[i] = self.nums[s]
                    self.sum[i] = self.nums[s]
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val
                    self.left[i] = val
                    self.right[i] = val
                    self.sum[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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
        if self.lazy[i] != inf:
            self.make_tag(s, m, 2 * i, self.lazy[i])
            self.make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = inf

    def build(self) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = self.nums[s]
                    self.left[i] = self.nums[s]
                    self.right[i] = self.nums[s]
                    self.sum[i] = self.nums[s]
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val if val < 0 else val*(t-s+1)
                    self.left[i] = val if val < 0 else val*(t-s+1)
                    self.right[i] = val if val < 0 else val*(t-s+1)
                    self.sum[i] = val*(t-s+1)
                    self.lazy[i] = val
                    continue
                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间的最大连续和，注意这里是递归
        if left <= s and t <= right:
            return [self.cover[i], self.left[i], self.right[i], self.sum[i]]

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
        highest = inf
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur < highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur < highest:
                highest = cur
        return highest


class SegmentTreeRangeUpdateQuery:
    def __init__(self, n) -> None:
        # 模板：区间修改、单点值查询
        self.n = n
        self.lazy = [0] * (4 * self.n)
        return

    def push_down(self, i):
        if self.lazy[i]:
            self.lazy[2*i] = self.lazy[2*i+1] = self.lazy[i]
            self.lazy[i] = 0

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 left == right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                self.lazy[i] = val
                continue
            m = s + (t - s) // 2
            self.push_down(i)
            if left <= m:  # 注意左右子树的边界与范围
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return

    def query(self, left: int, right: int, s: int, t: int, i: int):
        # 查询单点值
        while not (left <= s and t <= right):
            m = s + (t - s) // 2
            self.push_down(i)
            if right <= m:
                s, t, i = s, m, 2 * i
            elif left > m:
                s, t, i = m + 1, t, 2 * i + 1
        return self.lazy[i]


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
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.cover_2[i] = nums[s]*nums[s]
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增加 val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(s, t, i, val)
                    continue
                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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
        # 模板：区间修改成01或者翻转，区间查询最多有多少连续的1，以及总共有多少1
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


class SegmentTreeLongestSubSame:
    # 模板：单点字母更新，最长具有相同字母的连续子数组查询
    def __init__(self, n, lst):
        self.n = n
        self.lst = lst
        self.pref = [0] * 4 * n
        self.suf = [0] * 4 * n
        self.ceil = [0] * 4 * n
        self.build()
        return

    def build(self):
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.make_tag(ind)
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind, s, t)
        return

    def make_tag(self, i):
        # 只有此时 i 对应的区间 s==t 才打标记
        self.pref[i] = 1
        self.suf[i] = 1
        self.ceil[i] = 1
        return

    def push_up(self, i, s, t):
        m = s + (t - s) // 2
        # 左右区间段分别为 [s, m] 与 [m+1, t] 保证 s < t
        self.pref[i] = self.pref[2 * i]
        if self.pref[2 * i] == m - s + 1 and self.lst[m] == self.lst[m + 1]:
            self.pref[i] += self.pref[2 * i + 1]

        self.suf[i] = self.suf[2 * i + 1]
        if self.suf[2 * i + 1] == t - m and self.lst[m] == self.lst[m + 1]:
            self.suf[i] += self.suf[2 * i]

        a = -inf
        for b in [self.pref[i], self.suf[i], self.ceil[2*i], self.ceil[2*i+1]]:
            a = a if a > b else b
        if self.lst[m] == self.lst[m + 1]:
            b = self.suf[2 * i] + self.pref[2 * i + 1]
            a = a if a > b else b
        self.ceil[i] = a
        return

    def update_point(self, left, right, s, t, val, i):
        # 更新单点最小值
        self.lst[left] = val
        stack = []
        while True:
            stack.append([s, t, i])
            if left <= s <= t <= right:
                self.make_tag(i)
                break
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        stack.pop()
        while stack:
            s, t, i = stack.pop()
            self.push_up(i, s, t)
        assert i == 1
        # 获取当前最大的连续子串
        return self.ceil[1]


class SegmentTreeRangeXORQuery:
    def __init__(self, n) -> None:
        # 模板：区间异或修改、单点异或查询
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    def push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.cover[2 * i] ^ self.cover[2 * i + 1]
        return

    def make_tag(self, i, val):
        self.cover[i] ^= val
        self.lazy[i] ^= val
        return

    def push_down(self, i):
        if self.lazy[i]:
            self.make_tag(2 * i, self.lazy[i])
            self.make_tag(2 * i + 1, self.lazy[i])
            self.lazy[i] = 0

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 异或 val 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                m = s + (t - s) // 2
                if left <= s and t <= right:
                    self.make_tag(i, val)
                    continue
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

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 异或 val 而 i 从 1 开始
        assert left == right
        while True:
            if left <= s and t <= right:
                self.make_tag(i, val)
                break
            self.push_down(i)
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
        return

    def query(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans ^= self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans

    def query_point(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        assert left == right
        ans = 0
        while True:
            if left <= s and t <= right:
                ans ^= self.cover[i]
                break
            m = s + (t - s) // 2
            self.push_down(i)
            if left <= m:
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        return ans


class SegmentTreeRangeSqrtSum:
    def __init__(self, n):
        # 模板：区间值开方向下取整，区间和查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [inf] * (4 * self.n)  # 懒标记

    def build(self, nums):
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.cover[ind] = nums[s]
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.cover[ind] = self.cover[2 * ind] + self.cover[2 * ind + 1]
        return

    def change(self, left, right, s, t, i):
        # 更新区间值
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if self.cover[i] == t-s+1:
                    continue
                if s == t:
                    self.cover[i] = int(self.cover[i]**0.5)
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
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
    def lc_2213(s: str, word: str, indices: List[int]) -> List[int]:
        # 模板：单点字母更新，最长具有相同字母的连续子数组查询
        n = len(s)
        tree = SegmentTreeLongestSubSame(n, [ord(w) - ord("a") for w in s])
        ans = []
        for i, w in zip(indices, word):
            ans.append(tree.update_point(i, i, 0, n - 1, ord(w) - ord("a"), 1))
        return ans

    @staticmethod
    def lc_2569_2(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # 模板：经典01线段树区间翻转与求和，也可以使用BitSet
        res = []
        seg = SegmentTreeBitSet()
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
    def lc_2569_1(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # 模板：经典01线段树区间翻转与求和，也可以使用BitSet
        n = len(nums1)
        tree = SegmentTreeRangeUpdateXORSum(n)
        tree.build(nums1)
        ans = []
        s = sum(nums2)
        for op, x, y in queries:
            if op == 1:
                tree.update_range(x, y, 0, n-1, 1, 1)
            elif op == 2:
                s += tree.query_sum(0, n-1, 0, n-1, 1)*x
            else:
                ans.append(s)
        return ans


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
        segment = SegmentTreeRangeUpdateQuerySumMinMax(n)
        segment.build(ac.read_list_ints())

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                segment.update_range(x-1, y-1, 0, n-1, k, 1)
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
                segment.update_range(x-1, y-1, 0, n-1, 1, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_sum(x-1, y-1, 0, n-1, 1))
        return

    @staticmethod
    def lg_p1438(ac=FastIO()):
        # 模板：差分数组区间增减加线段树查询区间和
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        segment = SegmentTreeRangeUpdateQuerySumMinMax(n)

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k, d = lst[1:]
                if x == y:
                    segment.update_range(x - 1, x - 1, 0, n - 1, k, 1)
                    if y <= n - 1:
                        segment.update_point(y, y, 0, n - 1, -k, 1)
                else:
                    # 经典使用差分数组进行区间的等差数列加减
                    segment.update_point(x - 1, x - 1, 0, n - 1, k, 1)
                    segment.update_range(x, y - 1, 0, n - 1, d, 1)
                    cnt = y-x
                    if y <= n - 1:
                        segment.update_point(y, y, 0, n - 1, -cnt*d-k, 1)
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
    def lg_p6627(ac=FastIO()):
        # 模板：线段树维护和查询区间异或值
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nodes = {0, -10**9-1, 10**9+1}
        for lst in nums:
            for va in lst[1:-1]:
                nodes.add(va)
                nodes.add(va-1)
                nodes.add(va+1)
        nodes = sorted(list(nodes))
        n = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        tree = SegmentTreeRangeXORQuery(n)
        arr = [0]*n
        for lst in nums:
            if lst[0] == 1:
                a, b, w = lst[1:]
                if a > b:
                    a, b = b, a
                tree.update_range(ind[a], ind[b], 0, n-1, w, 1)
            elif lst[0] == 2:
                a, w = lst[1:]
                arr[ind[a]] ^= w  # 使用数组代替
                # tree.update_point(ind[a], ind[a], 0, n-1, w, 1)
            else:
                a, w = lst[1:]
                tree.update_range(0, n-1, 0, n - 1, w, 1)
                arr[ind[a]] ^= w  # 使用数组代替
                # tree.update_point(ind[a], ind[a], 0, n - 1, w, 1)
        ans = inf
        res = -inf
        for i in range(n):
            val = tree.query_point(i, i, 0, n-1, 1) ^ arr[i]
            if val > res or (val == res and (abs(ans) > abs(nodes[i]) or (abs(ans) == abs(nodes[i]) and nodes[i] > ans))):
                res = val
                ans = nodes[i]
        ac.lst([res, ans])
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
        # 模板：区间修改成01或者翻转，区间查询最多有多少连续的1，以及总共有多少1
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
                # 修改区间值
                tree.update(a - 1, b - 1, 0, n - 1, 1 << (c - 1), 1)
            else:
                a, b = [int(w) for w in lst[1:]]
                if a > b:
                    a, b = b, a
                # 区间值或查询
                ac.st(bin(tree.query_or(a - 1, b - 1, 0, n - 1, 1)).count("1"))
        return

    @staticmethod
    def lg_p3740(ac=FastIO()):
        # 模板：离散化线段树区间修改与单点查询
        n, m = ac.read_ints()
        nums = []
        while len(nums) < m * 2:
            nums.extend(ac.read_list_ints())
        nums = [nums[2 * i:2 * i + 2] for i in range(m)]
        nodes = set()
        nodes.add(1)
        nodes.add(n)
        for a, b in nums:
            nodes.add(a)
            nodes.add(b)
            # 离散化特别注意需要增加右端点进行连续区间的区分
            nodes.add(b + 1)
        nodes = list(sorted(nodes))
        ind = {num: i for i, num in enumerate(nodes)}

        # 区间修改
        n = len(nodes)
        tree = SegmentTreeRangeUpdateQuery(n)
        for i in range(m):
            a, b = nums[i]
            tree.update(ind[a], ind[b], 0, n - 1, i + 1, 1)

        # 单点查询
        ans = set()
        for i in range(n):
            c = tree.query(i, i, 0, n - 1, 1)
            if c:
                ans.add(c)
        ac.st(len(ans))
        return

    @staticmethod
    def lg_p4588(ac=FastIO()):
        # 模板：转化为线段树单点值修改与区间乘积取模
        for _ in range(ac.read_int()):
            q, mod = ac.read_ints()
            tree = SegmentTreePointUpdateRangeMulQuery(q, mod)
            for i in range(q):
                op, num = ac.read_ints()
                if op == 1:
                    tree.update(i, i, 0, q - 1, num % mod, 1)
                else:
                    tree.update(num - 1, num - 1, 0, q - 1, 1, 1)
                ac.st(tree.cover[1])
        return

    @staticmethod
    def lg_p8081(ac=FastIO()):
        # 模板：线段树区间修改、区间加和查询
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = SegmentTreeRangeUpdateQuerySum(n)
        pre = 0
        ceil = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre:
                    ceil = max(ceil, pre)
                    low, high = i-3*pre, i-pre-1
                    if high >= 0:
                        tree.update_range(ac.max(0, low), high, 0, n-1, 1, 1)
                pre = 0
        if pre:
            ceil = max(ceil, pre)
            low, high = n - 3 * pre, n - pre - 1
            if high >= 0:
                tree.update_range(ac.max(0, low), high, 0, n - 1, 1, 1)

        ans = tree.query_range(0, n-1, 0, n-1, 1)
        pre = 0
        res = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre == ceil:
                    low, high = i-4*pre, i-3*pre-1
                    low = ac.max(low, 0)
                    if low <= high:
                        res = ac.max(res, high-low+1-tree.query_range(low, high, 0, n-1, 1))
                pre = 0
        if pre == ceil:
            low, high = n - 4 * pre, n - 3 * pre - 1
            low = ac.max(low, 0)
            if low <= high:
                res = ac.max(res, high - low + 1 - tree.query_range(low, high, 0, n - 1, 1))
        ac.st(ans + res)
        return

    @staticmethod
    def lg_p8812(ac=FastIO()):
        # 模板：线段树查询和更新区间最小值
        n, m = ac.read_ints()
        goods = [[] for _ in range(n)]
        for _ in range(m):
            s, t, p, c = ac.read_ints()
            for _ in range(c):
                a, b = ac.read_ints()
                a -= 1
                goods[a].append([1, 10**9 + 1, b])
                b = b * p // 100
                goods[a].append([s, t, b])

        for i in range(n):
            nodes = {0, 10**9 + 1}
            for s, t, _ in goods[i]:
                nodes.add(s - 1)
                nodes.add(s)
                nodes.add(t)
                nodes.add(t + 1)
            nodes = sorted(list(nodes))
            ind = {node: i for i, node in enumerate(nodes)}
            k = len(ind)
            tree = SegmentTreeUpdateQueryMin(k)
            for s, t, b in goods[i]:
                tree.update_range(ind[s], ind[t], 0, k - 1, b, 1)
            res = []
            for x in range(k):
                val = tree.query_point(x, x, 0, k - 1, 1)
                if val == inf:
                    continue
                if not res or res[-1][2] != val:
                    res.append([nodes[x], nodes[x], val])
                else:
                    res[-1][1] = nodes[x]

            goods[i] = [r[:] for r in res]

        nodes = {0, 10 ** 9 + 1}
        for i in range(n):
            for s, t, _ in goods[i]:
                nodes.add(s)
                nodes.add(t)
        nodes = sorted(list(nodes))
        ind = {node: i for i, node in enumerate(nodes)}
        k = len(ind)
        diff = [0] * k
        for i in range(n):
            for s, t, b in goods[i]:
                diff[ind[s]] += b
                if ind[t] + 1 < k:
                    diff[ind[t] + 1] -= b
        diff = ac.accumulate(diff)[2:]
        ac.st(min(diff))
        return

    @staticmethod
    def cf_987c(ac=FastIO()):
        # 模板：枚举中间数组，使用线段树维护前后缀最小值
        n = ac.read_int()
        s = ac.read_list_ints()
        c = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(sorted(list(set(s + c + [0] + [10 ** 9 + 1]))))}
        m = len(ind)
        post = [inf] * n
        tree = SegmentTreeUpdateQueryMin(m)
        for i in range(n - 1, -1, -1):
            tree.update_point(ind[s[i]], ind[s[i]], 0, m - 1, c[i], 1)
            post[i] = tree.query_range(ind[s[i]] + 1, m - 1, 0, m - 1, 1)

        ans = inf
        tree = SegmentTreeUpdateQueryMin(m)
        for i in range(n):
            if 1 <= i <= n - 2:
                cur = c[i] + tree.query_range(0, ind[s[i]] - 1, 0, m - 1, 1) + post[i]
                ans = ac.min(ans, cur)
            tree.update_point(ind[s[i]], ind[s[i]], 0, m - 1, c[i], 1)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lc_1851(intervals: List[List[int]], queries: List[int]) -> List[int]:
        # 模板：区间更新最小值、单点查询
        port = []
        for inter in intervals:
            port.extend(inter)
        port.extend(queries)
        lst = sorted(list(set(port)))

        ind = {num: i for i, num in enumerate(lst)}
        ceil = len(lst)
        tree = SegmentTreeUpdateQueryMin(ceil)
        for a, b in intervals:
            tree.update_range(ind[a], ind[b], 0, ceil-1, b - a + 1, 1)
        ans = [tree.query_point(ind[num], ind[num], 0, ceil-1, 1) for num in queries]
        return [x if x != inf else -1 for x in ans]

    @staticmethod
    def lc_1340(nums: List[int], d: int) -> int:

        # 模板：可以使用线段树DP进行解决
        n = len(nums)
        post = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                post[stack.pop()] = i - 1
            stack.append(i)

        pre = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                pre[stack.pop()] = i + 1
            stack.append(i)

        # 分桶排序转移
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        tree = SegmentTreeRangeAddMax(n)
        for num in sorted(dct):
            cur = []
            for i in dct[num]:
                left, right = pre[i], post[i]
                if left < i - d:
                    left = i - d
                if right > i + d:
                    right = i + d
                x = tree.query(left, right, 0, n - 1, 1)
                cur.append([x + 1, i])

            for x, i in cur:
                tree.update(i, i, 0, n - 1, x, 1)
        return tree.query(0, n - 1, 0, n - 1, 1)


class CountIntervalsLC2276:

    def __init__(self):
        # 模板：动态开点线段树
        self.n = 10**9 + 7
        self.tree = SegmentTreeRangeChangeQuerySumMinMaxDefaultDict()

    def add(self, left: int, right: int) -> None:
        self.tree.update_range(left, right, 0, self.n-1, 1, 1)

    def count(self) -> int:
        return self.tree.cover[1]

    @staticmethod
    def ac_3805(ac=FastIO()):
        # 模板：区间增减与最小值查询
        n = ac.read_int()
        tree = SegmentTreeRangeUpdateQuerySumMinMax(n)
        tree.build(ac.read_list_ints())
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                l, r = lst
                if l <= r:
                    ac.st(tree.query_min(l, r, 0, n-1, 1))
                else:
                    ans1 = tree.query_min(l, n-1, 0, n-1, 1)
                    ans2 = tree.query_min(0, r, 0, n-1, 1)
                    ac.st(ac.min(ans1, ans2))
            else:
                l, r, d = lst
                if l <= r:
                    tree.update_range(l, r, 0, n-1, d, 1)
                else:
                    tree.update_range(l, n-1, 0, n-1, d, 1)
                    tree.update_range(0, r, 0, n-1, d, 1)
        return

    @staticmethod
    def ac_5037_1(ac=FastIO()):
        # 模板：同CF242E，使用二十多个01线段树维护区间异或与区间加和
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegmentTreeRangeUpdateXORSum(n) for _ in range(22)]
        for j in range(22):
            lst = [1 if nums[i] & (1 << j) else 0 for i in range(n)]
            tree[j].build(lst)
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1<<j)*tree[j].query_sum(ll, rr, 0, n-1, 1) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1<<j) & xx:
                        tree[j].update(ll, rr, 0, n-1, 1, 1)

        return

    @staticmethod
    def ac_5037_2(ac=FastIO()):
        # 模板：同CF242E，使用二十多个01线段树维护区间异或与区间加和
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegmentTreeBitSet() for _ in range(22)]
        for i in range(n):
            x = nums[i]
            for j in range(22):
                if x & (1 << j):
                    tree[j].update(i, i)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1 << j)*tree[j].query(ll, rr) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1 << j) & xx:
                        tree[j].update(ll, rr)
        return


class BookMyShowLC2286:

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.tree = SegmentTreeRangeUpdateQuerySumMinMax(n)
        self.cnt = [0] * n
        self.null = SortedList(list(range(n)))

    def gather(self, k: int, max_row: int) -> List[int]:
        max_row += 1
        low = self.tree.query_min(0, max_row - 1, 0, self.n - 1, 1)
        if self.m - low < k:
            return []

        def check(x):
            return self.m - self.tree.query_min(0, x, 0, self.n - 1, 1) >= k

        # 模板：经典二分加线段树维护最小值与和
        y = BinarySearch().find_int_left(0, max_row - 1, check)
        self.cnt[y] += k
        self.tree.update_point(y, y, 0, self.n - 1, k, 1)
        if self.cnt[y] == self.m:
            self.null.discard(y)
        return [y, self.cnt[y] - k]

    def scatter(self, k: int, max_row: int) -> bool:
        max_row += 1
        s = self.tree.query_sum(0, max_row - 1, 0, self.n - 1, 1)
        if self.m * max_row - s < k:
            return False
        while k:
            x = self.null[0]
            rest = k if k < self.m - self.cnt[x] else self.m - self.cnt[x]
            k -= rest
            self.cnt[x] += rest
            self.tree.update_point(x, x, 0, self.n - 1, rest, 1)
            if self.cnt[x] == self.m:
                self.null.pop(0)
        return True
    
    
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

    def test_segment_tree_update_query_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeUpdateQueryMin(high)
        stra.build(nums)

        for _ in range(high):
            # 区间更新与查询最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update_range(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_range(left, right, low, high-1, 1) == min(
                nums[left:right + 1])

            # 单点更新与查询最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update_point(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert stra.query_point(left, right, low, high-1, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_range(left, right, low, high-1, 1) == min(
                nums[left:right + 1])

        assert stra.get() == nums[:]
        return

    def test_segment_tree_range_change_query_sum_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        n = len(nums)
        stra = SegmentTreeRangeUpdateQuerySumMinMax(n)
        stra.build(nums)
        for i in range(high):
            assert stra.query_min(i, i, low, high-1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high-1, 1) == sum(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update_range(left, right, low, high-1, num, 1)
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
            stra.update_point(left, right, low, high-1, num, 1)
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

        assert stra.get_all_nums() == nums
        return

    def test_segment_tree_range_update_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateQuerySumMinMax(len(nums))
        stra.build(nums)

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
            stra.update_range(left, right, low, high-1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high-1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high -1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值使用 update
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.update_range(left, right, low, high - 1, num, 1)
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

            # 单点更新最小值使用 update_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.update_point(left, right, low, high - 1, num, 1)
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
        assert stra.get_all_nums() == nums
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

            # 单点更新最小值 change_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.change_point(left, right, low, high - 1, num, 1)
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

        stra.get_all_nums()
        assert stra.nums == nums
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

    def test_segment_tree_range_change_point_query(self):
        low = 0
        high = 10**9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeUpdateQuery(n)
        for i in range(n):
            stra.update(i, i, 0, n-1, nums[i], 1)

        for _ in range(10):
            # 区间修改值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, 0, n-1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            for i in range(n):
                assert stra.query(i, i, 0, n-1, 1) == nums[i]
        return


if __name__ == '__main__':
    unittest.main()
