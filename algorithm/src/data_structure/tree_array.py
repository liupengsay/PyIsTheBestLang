import random
import unittest

"""
算法：树状数组
功能：进行数组区间加减，和区间值求和（单点可转换为区间）
题目：
P3374 树状数组 1（https://www.luogu.com.cn/problem/P3374）区间值更新与求和
P3368 树状数组 2（https://www.luogu.com.cn/problem/P3368）区间值更新与求和
P5677 配对统计（https://www.luogu.com.cn/problem/P5677）区间值更新与求和
L2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数

P2068 统计和（https://www.luogu.com.cn/problem/P2068）单点更新与区间求和
P2345 [USACO04OPEN] MooFest G（https://www.luogu.com.cn/problem/P2345）使用两个树状数组计数与加和更新查询
P2357 守墓人（https://www.luogu.com.cn/problem/P2357）区间更新与区间求和
P2781 传教（https://www.luogu.com.cn/problem/P2781）区间更新与区间求和
P5200 [USACO19JAN]Sleepy Cow Sorting G（https://www.luogu.com.cn/problem/P5200）树状数组加贪心模拟

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


class TreeArrayPrefixMin:
    # 单点更新与前缀最小值
    def __init__(self, n):
        self.n = n
        # 数组索引从1开始
        self.c = [float("inf")]*(n+1)

    # 求x的二进制表示中，最低位的1的位置对应的数，向右相加更新管辖值，向左相减获得前缀和
    @staticmethod
    def lowest_bit(x):
        return x & -x

    # 给nums索引x增加k，同时维护对应受到影响的区间和c数组
    def add(self, x, k):
        while x <= self.n:  # 不能越界
            self.c[x] = min(self.c[x], k)
            x = x + self.lowest_bit(x)
        return

    # 前缀求最小值
    def get_prefix_min(self, x):  # a[1]..a[x]的最小值
        ans = float("inf")
        while x >= 1:
            ans = min(ans, self.c[x])
            x -= self.lowest_bit(x)
        return ans


class TreeArrayPrefixMax:
    # 单点更新与前缀最大值
    def __init__(self, n):
        self.n = n
        # 数组索引从1开始
        self.c = [float("-inf")]*(n+1)

    # 求x的二进制表示中，最低位的1的位置对应的数，向右相加更新管辖值，向左相减获得前缀和
    @staticmethod
    def lowest_bit(x):
        return x & -x

    # 给nums索引x增加k，同时维护对应受到影响的区间和c数组
    def add(self, x, k):
        while x <= self.n:  # 不能越界
            self.c[x] = max(self.c[x], k)
            x = x + self.lowest_bit(x)
        return

    # 前缀求最小值
    def get_prefix_min(self, x):  # a[1]..a[x]的最大值
        ans = float("inf")
        while x >= 1:
            ans = max(ans, self.c[x])
            x -= self.lowest_bit(x)
        return ans


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

    @unittest.skip
    def test_tree_array_prefix_min(self):
        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tapm = TreeArrayPrefixMin(ceil)
        for i in range(ceil):
            tapm.add(i+1, nums[i])

        for _ in range(ceil):
            d = random.randint(-ceil, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] += d
            tapm.add(i + 1, d)
            right = random.randint(0, ceil - 1)
            assert min(nums[:right + 1]) == tapm.get_prefix_min(right + 1)
        return

    @unittest.skip
    def test_tree_array_prefix_max(self):
        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(ceil)]
        tapm = TreeArrayPrefixMax(ceil)
        for i in range(ceil):
            tapm.add(i+1, nums[i])

        for _ in range(ceil):
            d = random.randint(-ceil, ceil)
            i = random.randint(0, ceil - 1)
            nums[i] += d
            tapm.add(i + 1, d)
            right = random.randint(0, ceil - 1)
            assert max(nums[:right + 1]) == tapm.get_prefix_min(right + 1)
        return


if __name__ == '__main__':
    unittest.main()
