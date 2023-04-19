import random
import unittest
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：树状数组
功能：进行数组区间加减，和区间值求和（单点可转换为区间）
题目：

===================================力扣===================================
1626. 无矛盾的最佳球队（https://leetcode.cn/problems/best-team-with-no-conflicts/）树状数组维护前缀最大值，也可使用动态规划求解
6353. 网格图中最少访问的格子数（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）树状数组维护前缀区间最小值单点更新

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

================================CodeForces================================
F. Range Update Point Query（https://codeforces.com/problemset/problem/1791/F）树状数组维护区间操作数与查询单点值
H2. Maximum Crossings (Hard Version)（https://codeforces.com/contest/1676/problem/H2）树状数组维护前缀区间和

参考：OI WiKi（https://oi-wiki.org/ds/fenwick/）
"""


class TreeArrayRangeSum:
    # 使用差分数组进行区间更新与求和
    def __init__(self, n):
        self.n = n
        # 数组索引从 1 开始
        self.t1 = [0] * (n + 1)
        self.t2 = [0] * (n + 1)

    @staticmethod
    def lowest_bit(x):
        return x & (-x)

    # 更新单点的差分数值
    def _add(self, k, v):
        v1 = k * v
        while k <= self.n:
            self.t1[k] = self.t1[k] + v
            self.t2[k] = self.t2[k] + v1
            k = k + self.lowest_bit(k)

    # 求差分数组的前缀和
    def _sum(self, t, k):
        ret = 0
        while k:
            ret = ret + t[k]
            k = k - self.lowest_bit(k)
        return ret

    # 更新差分的区间数值
    def update_range(self, l, r, v):
        self._add(l, v)
        self._add(r + 1, -v)

    # 求数组的前缀区间和
    def get_sum_range(self, l, r):
        a = (r + 1) * self._sum(self.t1, r) - self._sum(self.t2, r)
        b = l * self._sum(self.t1, l - 1) - self._sum(self.t2, l - 1)
        return a - b


class TreeArrayRangeQueryPointUpdateMax:
    # 模板：树状数组 单点更新 前缀区间查询 最大值
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


class TreeArrayRangeQuerySum:
    # 模板：树状数组 单点更新增减 查询前缀区间和
    def __init__(self, n):
        # 索引从 1 到 n
        self.t = [0] * (n + 1)

    @staticmethod
    def lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mi = 0
        while i:
            mi += self.t[i]
            i -= self.lowest_bit(i)
        return mi

    def update(self, i, mi):
        while i < len(self.t):
            self.t[i] += mi
            i += self.lowest_bit(i)
        return


class TreeArrayPointUpdateRangeMaxMin:

    # 模板：树状数组 单点更新修改 区间查询最大值与最小值
    def __init__(self, n: int) -> None:
        self.n = n
        self.a = [0] * (n + 1)
        self.tree_ceil = [0] * (n + 1)
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


class TreeArrayRangeQueryPointUpdateMin:
    # 模板：树状数组 单点更新修改 前缀区间查询最小值
    def __init__(self, n):
        # 索引从 1 到 n
        self.inf = float("inf")
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
        dp = [[float("inf")] * m for _ in range(n)]
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

        # 只能持续增加值
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

        # 只能持续减少值
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



if __name__ == '__main__':
    unittest.main()
