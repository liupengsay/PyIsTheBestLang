import math
import random
import unittest
import bisect
from collections import defaultdict
from functools import reduce

from algorithm.src.fast_io import FastIO

"""
算法：ST（Sparse-Table）稀疏表
功能：计算静态区间内的最大值、最小值、最大公约数、最大与、最大或
ST表算法全称Sparse-Table算法，是由Tarjan提出的一种解决RMQ问题（区间最值）的强力算法。 离线预处理时间复杂度θ（nlogn），在线查询时间θ（1），可以说是一种非常高效的算法。 不过ST表的应用场合也是有限的，它只能处理静态区间最值，不能维护动态的，也就是说不支持在预处理后对值进行修改。

题目：

===================================洛谷===================================
P3865 ST 表（https://www.luogu.com.cn/problem/P3865）使用ST表静态查询区间最大值
P2880 Balanced Lineup G（https://www.luogu.com.cn/problem/P2880）使用ST表预处理区间最大值与最小值
P1890 gcd区间（https://www.luogu.com.cn/problem/P3865）使用ST表预处理区间的gcd
P1816 忠诚（https://www.luogu.com.cn/problem/P1816）使用ST表预处理区间的最小值
P2412 查单词（https://www.luogu.com.cn/problem/P2412）预处理字典序之后使用ST表查询静态区间最大字典序
P2880 [USACO07JAN] Balanced Lineup G（https://www.luogu.com.cn/problem/P2880）查询区间最大值与最小值
P5097 [USACO04OPEN]Cave Cows 2（https://www.luogu.com.cn/problem/P5097）静态区间最小值

================================CodeForces================================
D. Max GEQ Sum（https://codeforces.com/problemset/problem/1691/D）单调栈枚举加ST表最大值最小值查询
D. Friends and Subsequences（https://codeforces.com/problemset/problem/689/D）根据单调性使用二分加ST表进行个数计算
D. Yet Another Yet Another Task（https://codeforces.com/problemset/problem/1359/D）单调栈枚举加ST表最大值最小值查询
B. Integers Have Friends（https://codeforces.com/problemset/problem/1548/B）ST表查询区间gcd并枚举数组开头，二分确定长度
474F（https://codeforces.com/problemset/problem/474/F）稀疏表计算最小值和gcd，并使用二分查找计数

参考：OI WiKi（xx）
"""


class SparseTable1:
    def __init__(self, lst, fun="max"):
        # 只要fun满足单调性就可以进行静态区间查询
        self.fun = fun
        self.n = len(lst)
        self.lst = lst
        self.f = [[0] * (int(math.log2(self.n)) + 1) for _ in range(self.n+1)]
        self.gen_sparse_table()

        return

    def gen_sparse_table(self):
        for i in range(1, self.n + 1):
            self.f[i][0] = self.lst[i - 1]
        for j in range(1, int(math.log2(self.n)) + 1):
            for i in range(1, self.n - (1 << j) + 2):
                a = self.f[i][j - 1]
                b = self.f[i + (1 << (j - 1))][j - 1]
                if self.fun == "max":
                    self.f[i][j] = a if a > b else b
                elif self.fun == "min":
                    self.f[i][j] = a if a < b else b
                elif self.fun == "gcd":
                    self.f[i][j] = math.gcd(a, b)
                elif self.fun == "lcm":
                    self.f[i][j] = a*b//math.gcd(a, b)
                elif self.fun == "and":
                    self.f[i][j] = a & b
                elif self.fun == "or":
                    self.f[i][j] = a | b

        return

    def query(self, left, right):
        # 查询数组的索引 left 和 right 从 1 开始
        k = int(math.log2(right - left + 1))
        a = self.f[left][k]
        b = self.f[right - (1 << k) + 1][k]
        if self.fun == "max":
            return a if a > b else b
        elif self.fun == "min":
            return a if a < b else b
        elif self.fun == "gcd":
            return math.gcd(a, b)
        elif self.fun == "lcm":
            return a * b // math.gcd(a, b)
        elif self.fun == "and":
            return a & b
        elif self.fun == "or":
            return a | b


class SparseTable2:
    def __init__(self, data, fun="max"):
        self.note = [0] * (len(data) + 1)
        self.fun = fun
        left, right, v = 1, 2, 0
        while True:
            for i in range(left, right):
                if i >= len(self.note):
                    break
                self.note[i] = v
            else:
                left *= 2
                right *= 2
                v += 1
                continue
            break
        self.ST = [[0] * len(data) for _ in range(self.note[-1]+1)]
        self.ST[0] = data
        for i in range(1, len(self.ST)):
            for j in range(len(data) - (1 << i) + 1):
                a, b = self.ST[i-1][j], self.ST[i-1][j + (1 << (i-1))]
                if self.fun == "max":
                    self.ST[i][j] = a if a > b else b
                elif self.fun == "min":
                    self.ST[i][j] = a if a < b else b
                else:
                    self.ST[i][j] = math.gcd(a, b)
        return

    def query(self, left, right):
        # 查询数组的索引 left 和 right 从 0 开始
        pos = self.note[right-left+1]
        a, b = self.ST[pos][left], self.ST[pos][right - (1 << pos) + 1]
        if self.fun == "max":
            return a if a > b else b
        elif self.fun == "min":
            return a if a < b else b
        else:
            return math.gcd(a, b)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2880(ac=FastIO()):
        # 模板：查询静态区间最大值与最小值
        n, q = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]
        st1 = SparseTable1(nums, "max")
        st2 = SparseTable1(nums, "min")
        for _ in range(q):
            a, b = ac.read_ints()
            ac.st(st1.query(a, b)-st2.query(a, b))
        return

    @staticmethod
    def lg_p3865(ac=FastIO()):
        # 模板：查询静态区间最大值
        n, m = ac.read_ints()
        st = SparseTable1(ac.read_list_ints())
        for _ in range(m):
            x, y = ac.read_ints()
            ac.st(st.query(x, y))
        return

    @staticmethod
    def cf_474f(ac=FastIO()):
        # 模板：使用稀疏表查询静态区间 gcd 与最小值
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        st_gcd = SparseTable1(nums, "gcd")
        st_min = SparseTable1(nums, "min")
        for _ in range(ac.read_int()):
            x, y = ac.read_ints_minus_one()
            num1 = st_gcd.query(x + 1, y + 1)
            num2 = st_min.query(x + 1, y + 1)
            if num1 == num2:
                res = bisect.bisect_right(dct[num1], y) - bisect.bisect_left(dct[num1], x)
                ac.st(y - x + 1 - res)
            else:
                ac.st(y - x + 1)
        return


class TestGeneral(unittest.TestCase):

    def test_sparse_table(self):

        def check_and(lst):
            ans = lst[0]
            for num in lst[1:]:
                ans &= num
            return ans

        def check_or(lst):
            ans = lst[0]
            for num in lst[1:]:
                ans |= num
            return ans

        nums = [9, 3, 1, 7, 5, 6, 0, 8]
        st = SparseTable1(nums)
        queries = [[1, 6], [1, 5], [2, 7], [2, 6], [1, 8], [4, 8], [3, 7], [1, 8]]
        assert [st.query(left, right) for left, right in queries] == [9, 9, 7, 7, 9, 8, 7, 9]

        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(1000)]
        st1_max = SparseTable1(nums, "max")
        st1_min = SparseTable1(nums, "min")
        st1_gcd = SparseTable1(nums, "gcd")
        st1_lcm = SparseTable1(nums, "lcm")
        st1_and = SparseTable1(nums, "and")
        st1_or = SparseTable1(nums, "or")

        st2_max = SparseTable2(nums, "max")
        st2_min = SparseTable2(nums, "min")
        st2_gcd = SparseTable2(nums, "gcd")
        for _ in range(ceil):
            left = random.randint(1, ceil-10)
            right = random.randint(left, ceil)
            assert st1_max.query(left, right) == st2_max.query(left-1, right-1) == max(nums[left-1:right])
            assert st1_min.query(left, right) == st2_min.query(left - 1, right - 1) == min(nums[left - 1:right])
            assert st1_gcd.query(left, right) == st2_gcd.query(left-1, right-1) == reduce(math.gcd, nums[left - 1:right])
            assert st1_lcm.query(left, right) == reduce(math.lcm, nums[left - 1:right])
            assert st1_and.query(left, right) == check_and(nums[left - 1:right])
            assert st1_or.query(left, right) == check_or(nums[left - 1:right])
        return


if __name__ == '__main__':
    unittest.main()
