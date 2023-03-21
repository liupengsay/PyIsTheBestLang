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

===================================洛谷===================================
P2068 统计和（https://www.luogu.com.cn/problem/P2068）单点更新与区间求和
P2345 [USACO04OPEN] MooFest G（https://www.luogu.com.cn/problem/P2345）使用两个树状数组计数与加和更新查询
P2357 守墓人（https://www.luogu.com.cn/problem/P2357）区间更新与区间求和
P2781 传教（https://www.luogu.com.cn/problem/P2781）区间更新与区间求和
P5200 [USACO19JAN]Sleepy Cow Sorting G（https://www.luogu.com.cn/problem/P5200）树状数组加贪心模拟
P3374 树状数组 1（https://www.luogu.com.cn/problem/P3374）区间值更新与求和
P3368 树状数组 2（https://www.luogu.com.cn/problem/P3368）区间值更新与求和
P5677 配对统计（https://www.luogu.com.cn/problem/P5677）区间值更新与求和


================================CodeForces================================
F. Range Update Point Query（https://codeforces.com/problemset/problem/1791/F）树状数组维护区间操作数与查询单点值


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


class TreeArrayPrefixMax:
    # 模板：树状数组维护数组前缀最大值
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


class Solution:
    def __init__(self):
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
        tree_array = TreeArrayPrefixMax(n)
        for score, age in sorted(zip(scores, ages)):
            cur = tree_array.query(age) + score
            tree_array.update(age, cur)
        return tree_array.query(n)


class TestGeneral(unittest.TestCase):

    def test_tree_array_range_sum(self):

        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tars = TreeArrayRangeSum(ceil)
        for i in range(ceil):
            tars.update_range(i + 1, i + 1, nums[i])

        for _ in range(ceil):
            d = random.randint(-ceil, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] += d
            tars.update_range(i + 1, i + 1, d)

            left = random.randint(0, ceil - 1)
            right = random.randint(left, ceil - 1)
            assert sum(nums[left: right + 1]) == tars.get_sum_range(left + 1, right + 1)


if __name__ == '__main__':
    unittest.main()
