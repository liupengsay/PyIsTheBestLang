import random
import unittest
from collections import defaultdict
from types import GeneratorType
from typing import List

from algorithm.src.fast_io import inf

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
P3372 线段树（https://www.luogu.com.cn/problem/P3372）区间值增减与计算区间和
P2846 [USACO08NOV]Light Switching G（https://www.luogu.com.cn/problem/P2846）线段树统计区间翻转和
P2574 XOR的艺术（https://www.luogu.com.cn/problem/P2574）线段树统计区间翻转和
P3130 [USACO15DEC] Counting Haybale P（https://www.luogu.com.cn/problem/P3130）区间增减、区间最小值查询、区间和查询
P3870 [TJOI2009] 开关（https://www.luogu.com.cn/problem/P3870） 区间值01翻转与区间和查询
P5057 [CQOI2006] 简单题（https://www.luogu.com.cn/problem/P5057） 区间值01翻转与区间和查询
P3372 【模板】线段树 1（https://www.luogu.com.cn/problem/P3372）区间值增减与求和

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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def bootstrap(f, queue=[]):
        def wrappedfunc(*args, **kwargs):
            if queue:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        queue.append(to)
                        to = next(to)
                    else:
                        queue.pop()
                        if not queue:
                            break
                        to = queue[-1].send(to)
                return to
        return wrappedfunc

    def cf_380c(self, word, quiries):
        # 模板：线段树进行分治并使用dp合并
        n = len(word)
        a = [0] * (4 * n)
        b = [0] * (4 * n)
        c = [0] * (4 * n)

        @self.bootstrap
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

        @self.bootstrap
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
        for x, y in quiries:
            d = defaultdict(list)
            query(x, y, 1, n, 1)
            ans.append(d[1][0])
        return ans


class SegmentTreeOrUpdateAndQuery:
    def __init__(self):
        # 区间按位或赋值、按位与查询
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
    def __init__(self):
        # 区间值01翻转与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2 * i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意使用异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意使用异或抵消查询

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] = t - s + 1 - self.cover[i]
            self.lazy[i] ^= val  # 注意使用异或抵消查询
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


class SegmentTreeRangeAddSum:
    def __init__(self):
        # 区间值增减与区间和查询
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
        # 区间值增减与区间和查询
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
        # 区间值修改与区间和查询
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


class SegmentTreeRangeAddMax:
    # 持续增加最大值
    def __init__(self, n):
        self.height = [0]*(4*n)
        self.lazy = [0]*(4*n)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.height[2 * i] if self.height[2 * i] > self.lazy[i] else self.lazy[i]
            self.height[2 * i + 1] = self.height[2 * i + 1] if self.height[2 * i + 1] > self.lazy[i] else self.lazy[i]

            self.lazy[2 * i] = self.lazy[2 * i] if self.lazy[2 * i] > self.lazy[i] else self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[2 * i + 1] if self.lazy[2 * i + 1] > self.lazy[i] else self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = self.height[i] if self.height[i] > val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] > val else val
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


class SegmentTreeRangeUpdateMax:
    # 持续修改区间值
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
        self.height[i] = self.height[2 * i] if self.height[2 *
                                                           i] > self.height[2 * i + 1] else self.height[2 * i + 1]
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


class SegmentTreeRangeUpdateMin:
    # 持续修改区间值
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
        self.height[i] = self.height[2 * i] if self.height[2 *
                                                           i] < self.height[2 * i + 1] else self.height[2 * i + 1]
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


class SegmentTreeRangeAddSumQueryMin:
    def __init__(self):
        # 区间值增加、区间和查询、区间最小值查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)
        self.floor = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] += self.lazy[i] * (t - m)

            self.floor[2 * i] += self.lazy[i]
            self.floor[2 * i + 1] += self.lazy[i]

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] += val * (t - s + 1)
            self.floor[i] += val
            self.lazy[i] += val
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]

        a, b = self.floor[2 * i], self.floor[2 * i + 1]
        self.floor[i] = a if a < b else b
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

    def query_min(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.floor[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = inf
        if left <= m:
            b = self.query_min(left, r, s, m, 2 * i)
            ans = ans if ans < b else b
        if r > m:
            b = self.query_min(left, r, m + 1, t, 2 * i + 1)
            ans = ans if ans < b else b
        return ans


class SegmentTreeRangeAddMin:
    def __init__(self):
        # 持续减小最小值
        self.height = defaultdict(lambda: float("inf"))
        self.lazy = defaultdict(lambda: float("inf"))

    def push_down(self, i):
        # 懒标记下放，注意取最小值
        if self.lazy[i] < float("inf"):
            self.height[2 *
                        i] = self.height[2 *
                                         i] if self.height[2 *
                                                           i] < self.lazy[i] else self.lazy[i]
            self.height[2 *
                        i +
                        1] = self.height[2 *
                                         i +
                                         1] if self.height[2 *
                                                           i +
                                                           1] < self.lazy[i] else self.lazy[i]

            self.lazy[2 * i] = self.lazy[2 * i] if self.lazy[2 *
                                                             i] < self.lazy[i] else self.lazy[i]
            self.lazy[2 *
                      i +
                      1] = self.lazy[2 *
                                     i +
                                     1] if self.lazy[2 *
                                                     i +
                                                     1] < self.lazy[i] else self.lazy[i]

            self.lazy[i] = float("inf")
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最小值
        if l <= s and t <= r:
            self.height[i] = self.height[i] if self.height[i] < val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] < val else val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 *
                                                           i] < self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最小值
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
        return highest if highest < float("inf") else -1


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


class TestGeneral(unittest.TestCase):

    def test_segment_tree_range_add_sum(self):
        low = 0
        high = 10**9 + 7
        n = 1000
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
        n = 1000
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
        high = 1000

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
        high = 1000
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
        high = 1000
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
        high = 1000
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
        high = 1000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeAddMin()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])
        return


if __name__ == '__main__':
    unittest.main()
