import random
import unittest
from math import inf
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：树状数组、二维树状数组
功能：进行数组区间加减，和区间值求和（单点可转换为区间）
题目：

===================================力扣===================================
1626. 无矛盾的最佳球队（https://leetcode.cn/problems/best-team-with-no-conflicts/）树状数组维护前缀最大值，也可使用动态规划求解
6353. 网格图中最少访问的格子数（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）树状数组维护前缀区间最小值单点更新
308. 二维区域和检索 - 可变（https://leetcode.cn/problems/range-sum-query-2d-mutable/）二维树状数组，单点增减与区间和查询


===================================洛谷===================================
P2068 统计和（https://www.luogu.com.cn/problem/P2068）单点更新与区间求和
P2345 [USACO04OPEN] MooFest G（https://www.luogu.com.cn/problem/P2345）使用两个树状数组计数与加和更新查询
P2357 守墓人（https://www.luogu.com.cn/problem/P2357）区间更新与区间求和
P2781 传教（https://www.luogu.com.cn/problem/P2781）区间更新与区间求和
P5200 [USACO19JAN]Sleepy Cow Sorting G（https://www.luogu.com.cn/problem/P5200）树状数组加贪心模拟
P3374 树状数组 1（https://www.luogu.com.cn/problem/P3374）区间值更新与求和
P3368 树状数组 2（https://www.luogu.com.cn/problem/P3368）区间值更新与求和
P5677 配对统计（https://www.luogu.com.cn/problem/P5677）区间值更新与求和
P5094 [USACO04OPEN] MooFest G 加强版（https://www.luogu.com.cn/problem/P5094）单点更新增加值与前缀区间和查询
P1816 忠诚（https://www.luogu.com.cn/problem/P1816）树状数组查询静态区间最小值
P1908 逆序对（https://www.luogu.com.cn/problem/P1908）树状数组求逆序对
P1725 琪露诺（https://www.luogu.com.cn/problem/P1725）倒序线性DP，单点更新值，查询区间最大值
P3586 [POI2015] LOG（https://www.luogu.com.cn/problem/P3586）离线查询、离散化树状数组，单点增减，前缀和查询
P1198 [JSOI2008] 最大数（https://www.luogu.com.cn/problem/P1198）树状数组，查询区间最大值
P4868 Preprefix sum（https://www.luogu.com.cn/problem/P4868）经典转换公式单点修改，使用两个树状数组维护前缀和的前缀和
P5463 小鱼比可爱（加强版）（https://www.luogu.com.cn/problem/P5463）经典使用树状数组维护前缀计数，枚举最大值计算所有区间数贡献

================================CodeForces================================
F. Range Update Point Query（https://codeforces.com/problemset/problem/1791/F）树状数组维护区间操作数与查询单点值
H2. Maximum Crossings (Hard Version)（https://codeforces.com/contest/1676/problem/H2）树状数组维护前缀区间和

135. 二维树状数组3（https://loj.ac/p/135）区间修改，区间查询
134. 二维树状数组2（https://loj.ac/p/134）区间修改，单点查询

参考：OI WiKi（https://oi-wiki.org/ds/fenwick/）
"""


class TreeArrayRangeQuerySum:
    # 模板：树状数组 单点增减 查询前缀和与区间和
    def __init__(self, n: int) -> None:
        # 索引从 1 到 n
        self.t = [0] * (n + 1)
        # 树状数组中每个位置保存的是其向前 low_bit 的区间和
        return

    def build(self, nums: List[int]) -> None:
        # 索引从 1 开始使用数组初始化树状数组
        n = len(nums)
        pre = [0]*(n+1)
        for i in range(n):
            pre[i+1] = pre[i] + nums[i]
            self.t[i+1] = pre[i+1] - pre[i+1-self.lowest_bit(i+1)]
        return

    @staticmethod
    def lowest_bit(i: int) -> int:
        # 经典 low_bit 即最后一位二进制为 1 所表示的数
        return i & (-i)

    def query(self, i: int) -> int:
        # 索引从 1 开始，查询 1 到 i 的前缀区间和
        mi = 0
        while i:
            mi += self.t[i]
            i -= self.lowest_bit(i)
        return mi

    def query_range(self, x: int, y: int) -> int:
        # 索引从 1 开始，查询 x 到 y 的值
        return self.query(y) - self.query(x-1)

    def update(self, i: int, mi: int) -> None:
        # 索引从 1 开始，索引 i 的值增加 mi 且 mi 可正可负
        while i < len(self.t):
            self.t[i] += mi
            i += self.lowest_bit(i)
        return


class TreeArrayRangeSum:
    # 模板：树状数组 区间增减 查询前缀和与区间和
    def __init__(self, n: int) -> None:
        self.n = n
        # 索引从 1 开始
        self.t1 = [0] * (n + 1)
        self.t2 = [0] * (n + 1)
        return

    def build(self, nums: List[int]) -> None:
        # 索引从 1 开始使用数组初始化树状数组
        n = len(nums)
        for i in range(n):
            self.update_range(i+1, i+1, nums[i])
        return

    @staticmethod
    def lowest_bit(x: int) -> int:
        # 经典 low_bit 即最后一位二进制为 1 所表示的数
        return x & (-x)

    # 更新单点的差分数值
    def _add(self, k: int, v: int) -> None:
        # 索引从 1 开始将第 k 个数加 v 且 v 可正可负
        v1 = k * v
        while k <= self.n:
            self.t1[k] = self.t1[k] + v
            self.t2[k] = self.t2[k] + v1
            k = k + self.lowest_bit(k)
        return

    # 求差分数组的前缀和
    def _sum(self, t: List[int], k: int) -> int:
        # 索引从 1 开始求前 k 个数的前缀和
        ret = 0
        while k:
            ret = ret + t[k]
            k = k - self.lowest_bit(k)
        return ret

    # 更新差分的区间数值
    def update_range(self, left: int, right: int, v: int) -> None:
        # 索引从 1 开始将区间 [left, right] 的数增加 v 且 v 可正可负
        self._add(left, v)
        self._add(right + 1, -v)
        return

    # 求数组的前缀区间和
    def get_sum_range(self, left: int, right: int) -> int:
        # 索引从 1 开始查询区间 [left, right] 的和
        a = (right + 1) * self._sum(self.t1, right) - self._sum(self.t2, right)
        b = left * self._sum(self.t1, left - 1) - self._sum(self.t2, left - 1)
        return a - b


class TreeArrayRangeQueryPointUpdateMax:
    # 模板：树状数组 单点增加 前缀区间查询 最大值
    def __init__(self, n):
        # 索引从 1 到 n
        self.t = [0] * (n + 1)

    @staticmethod
    def lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mx = 0
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self.lowest_bit(i)
        return mx

    def update(self, i, mx):
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self.lowest_bit(i)
        return


class TreeArrayRangeQueryPointUpdateMin:
    # 模板：树状数组 单点减少 前缀区间查询最小值
    def __init__(self, n):
        # 索引从 1 到 n
        self.inf = inf
        self.t = [self.inf] * (n + 1)

    @staticmethod
    def lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mi = self.inf
        while i:
            mi = mi if mi < self.t[i] else self.t[i]
            i -= self.lowest_bit(i)
        return mi

    def update(self, i, mi):
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < mi else mi
            i += self.lowest_bit(i)
        return


class TreeArrayPointUpdateRangeMaxMin:

    # 模板：树状数组 单点增加区间查询最大值 单点减少区间查询最小值
    def __init__(self, n: int) -> None:
        self.n = n
        self.a = [0] * (n + 1)
        self.tree_ceil = [-inf] * (n + 1)
        self.tree_floor = [float('inf')] * (n + 1)
        return

    @staticmethod
    def low_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def add(self, x, k):
        # 索引从1开始
        self.a[x] = k
        while x <= self.n:
            self.tree_ceil[x] = self.max(self.tree_ceil[x], k)
            self.tree_floor[x] = self.min(self.tree_floor[x], k)
            x += self.low_bit(x)
        return

    def find_max(self, left, r):
        # 索引从1开始
        max_val = float('-inf')
        while r >= left:
            if r - self.low_bit(r) >= left - 1:
                max_val = self.max(max_val, self.tree_ceil[r])
                r -= self.low_bit(r)
            else:
                max_val = self.max(max_val, self.a[r])
                r -= 1
        return max_val

    def find_min(self, left, r):
        # 索引从1开始
        min_val = float('inf')
        while r >= left:
            if r - self.low_bit(r) >= left - 1:
                min_val = self.min(min_val, self.tree_floor[r])
                r -= self.low_bit(r)
            else:
                min_val = self.min(min_val, self.a[r])
                r -= 1
        return min_val


class TreeArray2D:
    def __init__(self, m: int, n: int) -> None:
        # 模板：二维树状数组 单点增减 区间和查询
        self.m = m  # 行数
        self.n = n  # 列数
        self.tree = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        return

    def add(self, x: int, y: int, val: int) -> None:
        # 索引从 1 开始， 单点增加 val 到二维数组中坐标为 [x, y] 的值且 val 可正可负
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree[i][j] += val
                j += (j & -j)
            i += (i & -i)
        return

    def _query(self, x: int, y: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [1, 1] 到 [x, y] 的前缀和
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= (j & -j)
            i -= (i & -i)
        return res

    def range_query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [x1, y1] 到 [x2, y2] 的区间和
        return self._query(x2, y2) - self._query(x2, y1 - 1) - self._query(x1 - 1, y2) + self._query(x1 - 1, y1 - 1)


class TreeArray2DRange:
    def __init__(self, m: int, n: int) -> None:
        # 模板：二维树状数组 区间增减 区间和查询
        self.m = m  # 行数
        self.n = n  # 列数
        self.m = m
        self.n = n
        self.t1 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t2 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t3 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t4 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        return 

    def _add(self, x: int, y: int, val: int) -> None:
        # 索引从 1 开始， 单点增加 val 到二维数组中坐标为 [x, y] 的差分数组值且 val 可正可负
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.t1[i][j] += val
                self.t2[i][j] += val * x
                self.t3[i][j] += val * y
                self.t4[i][j] += val * x * y
                j += (j & -j)
            i += (i & -i)
        return

    def range_add(self, x1: int, y1: int, x2: int, y2: int, val: int) -> None:
        # 索引从 1 开始， 区间增加 val 到二维数组中坐标为左上角 [x1, y1] 到右下角的 [x2, y2] 且 val 可正可负
        self._add(x1, y1, val)
        self._add(x1, y2+1, -val)
        self._add(x2+1, y1, -val)
        self._add(x2+1, y2+1, val)
        return
    
    def _query(self, x: int, y: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [1, 1] 到 [x, y] 的前缀和
        res = 0
        i = x
        while i:
            j = y
            while j:
                res += (x + 1) * (y + 1) * self.t1[i][j] - (y + 1) * self.t2[i][j] - (x + 1) * self.t3[i][j] + self.t4[i][j]
                j -= (j & -j)
            i -= (i & -i)
        return res

    def range_query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [x1, y1] 到 [x2, y2] 的区间和
        return self._query(x2, y2) - self._query(x2, y1-1) - self._query(x1-1, y2) + self._query(x1-1, y1-1)


class TreeArray2DRangeMaxMin:
    # 模板：树状数组 单点增加区间查询最大值 单点减少区间查询最小值（暂未调通）
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self.a = [[0] * (n + 1) for _ in range(m + 1)]
        self.tree_ceil = [[0] * (n + 1) for _ in range(m + 1)]  # 最大值只能持续增加
        self.tree_floor = [[float('inf')] * (n + 1) for _ in range(m + 1)]  # 最小值只能持续减少
        return

    @staticmethod
    def low_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def add(self, x, y, k):
        # 索引从1开始
        self.a[x][y] = k
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree_ceil[i][j] = self.max(self.tree_ceil[i][j], k)
                self.tree_floor[i][j] = self.min(self.tree_floor[i][j], k)
                j += self.low_bit(j)
            i += self.low_bit(i)
        return

    def find_max(self, x1, y1, x2, y2):
        # 索引从1开始
        max_val = float('-inf')
        i1, i2 = x1, x2
        while i2 >= i1:
            if i2 - self.low_bit(i2) >= i1 - 1:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self.low_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self.low_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########

                i2 -= self.low_bit(i2)
            else:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self.low_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self.low_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########
                max_val = self.max(max_val, max(self.a[i2][y1:y2+1]))
                i2 -= 1
        return max_val


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_5094(ac=FastIO()):

        # 模板：树状数组单点增加值与前缀区间和查询
        n = ac.read_int()
        m = 5 * 10 ** 4

        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda y: y[0])
        tree_sum = TreeArrayRangeQuerySum(m)
        tree_cnt = TreeArrayRangeQuerySum(m)
        total_cnt = 0
        total_sum = 0
        ans = 0
        for v, x in nums:
            pre_sum = tree_sum.query(x)
            pre_cnt = tree_cnt.query(x)
            ans += v*(pre_cnt*x-pre_sum) + v*(total_sum-pre_sum-(total_cnt-pre_cnt)*x)
            tree_sum.update(x, x)
            tree_cnt.update(x, 1)
            total_cnt += 1
            total_sum += x
        ac.st(ans)
        return

    @staticmethod
    def lg_p2280(ac=FastIO()):
        # 模板：树状数组单点更新区间查询最大值与最小值
        n, q = ac.read_ints()
        tree = TreeArrayPointUpdateRangeMaxMin(n)
        for i in range(n):
            tree.a[i + 1] = ac.read_int()
            tree.add(i + 1, tree.a[i + 1])

        for _ in range(q):
            a, b = ac.read_ints()
            ac.st(tree.find_max(a, b) - tree.find_min(a, b))
        return

    @staticmethod
    def lc_6353(grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        dp = [[inf] * m for _ in range(n)]
        r, c = [TreeArrayRangeQueryPointUpdateMin(m) for _ in range(n)], [TreeArrayRangeQueryPointUpdateMin(n) for _ in range(m)]
        dp[n - 1][m - 1] = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if grid[i][j] > 0:
                    dp[i][j] = min(r[i].query(min(j + grid[i][j]+1, m)), c[j].query(min(i + grid[i][j]+1, n))) + 1
                if dp[i][j] <= n * m:
                    r[i].update(j+1, dp[i][j])
                    c[j].update(i+1, dp[i][j])
        return -1 if dp[0][0] > n * m else dp[0][0]

    @staticmethod
    def cf_1676h2(ac=FastIO()):
        # 模板：树状数组维护前缀区间和
        for _ in range(ac.read_int()):
            n = ac.read_int()
            a = ac.read_list_ints()
            ceil = max(a)
            ans = 0
            tree = TreeArrayRangeQuerySum(ceil)
            x = 0
            for num in a:
                ans += x - tree.query(num-1)
                tree.update(num, 1)
                x += 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p2068(ac=FastIO()):
        # 模板：树状数组单点更新与区间和查询
        n = ac.read_int()
        w = ac.read_int()
        tree = TreeArrayRangeSum(n)
        for _ in range(w):
            lst = ac.read_list_strs()
            a, b = int(lst[1]), int(lst[2])
            if lst[0] == "x":
                tree.update_range(a, a, b)
            else:
                ac.st(tree.get_sum_range(a, b))
        return

    @staticmethod
    def lc_1626(scores: List[int], ages: List[int]) -> int:
        # 模板：动态规划与树状数组维护前缀最大值
        n = max(ages)
        tree_array = TreeArrayRangeQueryPointUpdateMax(n)
        for score, age in sorted(zip(scores, ages)):
            cur = tree_array.query(age) + score
            tree_array.update(age, cur)
        return tree_array.query(n)

    @staticmethod
    def lg_p1816(ac=FastIO()):

        # 模板：树状数组查询静态区间最小值
        m, n = ac.read_ints()
        nums = ac.read_list_ints()
        tree = TreeArrayPointUpdateRangeMaxMin(m)
        for i in range(m):
            tree.add(i+1, nums[i])
        ans = []
        for _ in range(n):
            x, y = ac.read_ints()
            ans.append(tree.find_min(x, y))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p3374(ac=FastIO()):
        # 模板：树状数组 单点增减 查询前缀和与区间和
        n, m = ac.read_ints()
        tree = TreeArrayRangeQuerySum(n)
        tree.build(ac.read_list_ints())
        for _ in range(m):
            op, x, y = ac.read_ints()
            if op == 1:
                tree.update(x, y)
            else:
                ac.st(tree.query_range(x, y))
        return

    @staticmethod
    def lg_p3368(ac=FastIO()):
        # 模板：树状数组 区间增减 查询前缀和与区间和
        n, m = ac.read_ints()
        tree = TreeArrayRangeSum(n)
        tree.build(ac.read_list_ints())
        for _ in range(m):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                ac.st(tree.get_sum_range(lst[1], lst[1]))
            else:
                x, y, k = lst[1:]
                tree.update_range(x, y, k)
        return

    @staticmethod
    def lg_p1908(ac=FastIO()):
        # 模板：树状数组求逆序对
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = list(range(n))
        ind.sort(key=lambda x: nums[x])
        tree = TreeArrayRangeQuerySum(n)
        ans = i = cnt = 0
        while i < n:
            val = nums[ind[i]]
            lst = []
            while i < n and nums[ind[i]] == val:
                lst.append(ind[i]+1)
                ans += cnt - tree.query(ind[i]+1)
                i += 1
            cnt += len(lst)
            for x in lst:
                tree.update(x, 1)
        ac.st(ans)
        return

    @staticmethod
    def lc_308():
        class NumMatrix:
            def __init__(self, matrix: List[List[int]]):
                m, n = len(matrix), len(matrix[0])
                self.matrix = matrix
                self.tree = TreeArray2D(m, n)
                for i in range(m):
                    for j in range(n):
                        self.tree.add(i + 1, j + 1, matrix[i][j])

            def update(self, row: int, col: int, val: int) -> None:
                # 注意这里是修改为 val
                self.tree.add(row + 1, col + 1, val - self.matrix[row][col])
                self.matrix[row][col] = val

            def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
                return self.tree.range_query(row1 + 1, col1 + 1, row2 + 1, col2 + 1)
        return

    @staticmethod
    def main(ac=FastIO()):
        # 模板：二维树状数组 区间增减 区间查询
        n, m = ac.read_ints()
        tree = TreeArray2DRange(n, m)
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break
            if lst[0] == 1:
                a, b, c, d, x = lst[1:]
                tree.range_add(a, b, c, d, x)
            else:
                a, b, c, d = lst[1:]
                ac.st(tree.range_query(a, b, c, d))
        return

    @staticmethod
    def lg_p1725(ac=FastIO()):
        # 模板：树状数组倒序线性DP，单点更新与区间查询最大值
        n, a, b = ac.read_ints()
        n += 1
        nums = ac.read_list_ints()
        tree = TreeArrayPointUpdateRangeMaxMin(n+1)
        tree.add(n+1, 0)
        post = 0
        for i in range(n-1, -1, -1):
            x, y = i+a+1, i+b+1
            x = n+1 if x > n+1 else x
            y = n+1 if y > n+1 else y
            post = tree.find_max(x, y)
            tree.add(i+1, post+nums[i])
        ac.st(post+nums[0])
        return

    @staticmethod
    def lg_p3586(ac=FastIO()):
        # 模板：离线查询、离散化树状数组，单点增减，前缀和查询
        n, m = ac.read_ints()
        value = {0}
        lst = []
        for _ in range(m):
            cur = ac.read_list_strs()
            if cur[0] == "U":
                k, a = [int(w) for w in cur[1:]]
                value.add(a)
                lst.append([1, k, a])
            else:
                c, s = [int(w) for w in cur[1:]]
                value.add(s)
                lst.append([2, c, s])
        value = sorted(list(value))
        ind = {num: i for i, num in enumerate(value)}
        length = len(ind)

        tree_cnt = TreeArrayRangeQuerySum(length)
        tree_sum = TreeArrayRangeQuerySum(length)
        nums = [0]*n
        total_s = 0
        total_c = 0

        for op, a, b in lst:
            if op == 1:
                if nums[a-1]:
                    tree_cnt.update(ind[nums[a-1]], -1)
                    tree_sum.update(ind[nums[a - 1]], -nums[a-1])
                    total_s -= nums[a-1]
                    total_c -= 1
                nums[a-1] = b
                if nums[a - 1]:
                    tree_cnt.update(ind[nums[a - 1]], 1)
                    tree_sum.update(ind[nums[a - 1]], nums[a - 1])
                    total_s += nums[a-1]
                    total_c += 1
            else:
                c, s = a, b
                less_than_s = tree_cnt.query(ind[s]-1)
                less_than_s_sum = tree_sum.query(ind[s]-1)
                if (total_c-less_than_s)*s + less_than_s_sum >= c*s:
                    ac.st("TAK")
                else:
                    ac.st("NIE")
        return

    @staticmethod
    def lg_p1198(ac=FastIO()):
        # 模板：树状数组查询区间最大值
        m, d = ac.read_ints()
        t = 0
        tree = TreeArrayPointUpdateRangeMaxMin(m+1)
        i = 1
        for _ in range(m):
            op, x = ac.read_list_strs()
            if op == "A":
                x = (int(x)+t)%d
                tree.add(i, x)
                i += 1
            else:
                x = int(x)
                t = tree.find_max(i-x, i-1)
                ac.st(t)
        return

    @staticmethod
    def lg_p4868(ac=FastIO()):
        # 模板：经典转换公式，使用两个树状数组维护前缀和的前缀和

        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        tree1 = TreeArrayRangeQuerySum(n)
        tree1.build(nums)
        tree2 = TreeArrayRangeQuerySum(n)
        tree2.build([nums[i] * (i + 1) for i in range(n)])
        for _ in range(m):
            lst = ac.read_list_strs()
            if lst[0] == "Modify":
                i, x = [int(w) for w in lst[1:]]
                y = nums[i - 1]
                nums[i - 1] = x
                tree1.update(i, x - y)
                tree2.update(i, i * (x - y))
            else:
                i = int(lst[1])
                ac.st((i + 1) * tree1.query(i) - tree2.query(i))
        return

    @staticmethod
    def lg_p5463(ac=FastIO()):
        # 模板：经典使用树状数组维护前缀计数，枚举最大值计算所有区间数贡献
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = sorted(list(set(nums)))
        ind = {num: i + 1 for i, num in enumerate(lst)}
        m = len(ind)
        tree = TreeArrayRangeQuerySum(m)
        ans = 0
        for i in range(n - 1, -1, -1):
            left = i + 1
            right = tree.query(ind[nums[i]] - 1)
            ans += left * right
            # 取 nums[i] 作为区间的数又 n-i 个右端点取法
            tree.update(ind[nums[i]], n - i)
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_tree_array_range_sum(self):

        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tars = TreeArrayRangeSum(ceil)
        pm = TreeArrayRangeQueryPointUpdateMin(ceil)
        pm_max = TreeArrayRangeQueryPointUpdateMax(ceil)
        for i in range(ceil):
            tars.update_range(i + 1, i + 1, nums[i])
            pm.update(i+1, nums[i])
            pm_max.update(i+1, nums[i])
            assert pm.query(i + 1) == min(nums[:i + 1])
            assert pm_max.query(i+1) == max(nums[:i+1])

        for _ in range(ceil):
            d = random.randint(-ceil, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] += d
            tars.update_range(i + 1, i + 1, d)

            left = random.randint(0, ceil - 1)
            right = random.randint(left, ceil - 1)
            assert sum(nums[left: right + 1]) == tars.get_sum_range(left + 1, right + 1)

    def test_tree_array_range_max_min(self):

        # 只能持续增加值查询最大值
        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tree = TreeArrayPointUpdateRangeMaxMin(ceil)
        for i in range(ceil):
            tree.add(i+1, nums[i])
        for _ in range(ceil):
            d = random.randint(0, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] += d
            tree.add(i + 1, tree.a[i+1]+d)
            left = random.randint(0, ceil - 1)
            right = random.randint(left, ceil - 1)
            assert max(nums[left: right + 1]) == tree.find_max(left + 1, right + 1)

        # 只能持续减少值查询最小值
        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tree = TreeArrayPointUpdateRangeMaxMin(ceil)
        for i in range(ceil):
            tree.add(i+1, nums[i])

        for _ in range(ceil):
            d = random.randint(0, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] -= d
            tree.add(i + 1, tree.a[i + 1] - d)
            left = random.randint(0, ceil - 1)
            right = random.randint(left, ceil - 1)
            assert min(nums[left: right + 1]) == tree.find_min(left + 1, right + 1)

    def test_tree_array_2d_sum(self):

        # 二维树状数组，单点增减，区间查询
        m = n = 100
        high = 100000
        tree = TreeArray2D(m, n)
        grid = [[random.randint(-high, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.add(i+1, j+1, grid[i][j])
        for _ in range(m):
            row = random.randint(0, m-1)
            col = random.randint(0, n-1)
            x = random.randint(-high, high)
            grid[row][col] += x
            tree.add(row + 1, col + 1, x)
            x1 = random.randint(0, m-1)
            y1 = random.randint(0, n-1)
            x2 = random.randint(x1, m-1)
            y2 = random.randint(y1, n-1)
            assert tree.range_query(x1+1, y1+1, x2+1, y2+1) == sum(sum(g[y1:y2+1]) for g in grid[x1:x2+1])

        # 二维树状数组，区间增减，区间查询
        m = n = 100
        high = 100000
        tree = TreeArray2DRange(m, n)
        grid = [[random.randint(-high, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.range_add(i + 1, j + 1, i+1, j+1, grid[i][j])
        for _ in range(m):
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            x = random.randint(-high, high)
            for i in range(x1, x2+1):
                for j in range(y1, y2+1):
                    grid[i][j] += x

            tree.range_add(x1 + 1, y1 + 1, x2 + 1, y2 + 1, x)
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            assert tree.range_query(x1 + 1, y1 + 1, x2 + 1, y2 + 1) == sum(
                sum(g[y1:y2 + 1]) for g in grid[x1:x2 + 1])

    def test_tree_array_2d_max_min(self):

        # 二维树状数组，单点增减，区间查询
        random.seed(2023)
        m = n = 100
        high = 100000
        tree = TreeArray2DRangeMaxMin(m, n)
        grid = [[random.randint(0, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.add(i + 1, j + 1, grid[i][j])
        for _ in range(m):
            row = random.randint(0, m - 1)
            col = random.randint(0, n - 1)
            x = random.randint(0, high)
            grid[row][col] += x
            tree.add(row + 1, col + 1, grid[row][col])
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            ans1 = tree.find_max(x1 + 1, y1 + 1, x2 + 1, y2 + 1)
            ans2 = max(max(g[y1:y2 + 1]) for g in grid[x1:x2 + 1])
            print(ans1, ans2)
            assert ans1 == ans2
        return 
    
    
if __name__ == '__main__':
    unittest.main()
