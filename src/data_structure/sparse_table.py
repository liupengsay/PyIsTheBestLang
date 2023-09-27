import math
import random
import unittest
import bisect
from collections import defaultdict
from operator import or_, and_
from math import lcm, gcd
from functools import reduce
from typing import List
from math import inf

from src.fast_io import FastIO

"""
算法：ST（Sparse-Table）稀疏表、倍增、数组积性函数聚合性质、连续子数组的聚合运算
功能：计算静态区间内的最大值、最小值、最大公约数、最大与、最大或
ST表算法全称Sparse-Table算法，是由Tarjan提出的一种解决RMQ问题（区间最值）的强力算法。 离线预处理时间复杂度θ（nlogn），在线查询时间θ（1），可以说是一种非常高效的算法。 不过ST表的应用场合也是有限的，它只能处理静态区间最值，不能维护动态的，也就是说不支持在预处理后对值进行修改。

题目：

===================================力扣===================================
1521. 找到最接近目标值的函数值（https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/）经典计算与目标值最接近的连续子数组位运算与值
2411. 按位或最大的最小子数组长度（https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/）经典计算最大或值的最短连续子数组
2447. 最大公因数等于 K 的子数组数目（https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/）经典计算最大公因数为 k 的连续子数组个数，可推广到位运算或与异或
2470. 最小公倍数为 K 的子数组数目（https://leetcode.cn/problems/number-of-subarrays-with-lcm-equal-to-k/）经典计算最小公倍为 k 的连续子数组个数，可推广到位运算或与异或
2654. 使数组所有元素变成 1 的最少操作次数（https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）经典计算最大公因数为 1 的最短连续子数组

===================================洛谷===================================
P3865 ST 表（https://www.luogu.com.cn/problem/P3865）使用ST表静态查询区间最大值
P2880 Balanced Lineup G（https://www.luogu.com.cn/problem/P2880）使用ST表预处理区间最大值与最小值
P1890 gcd区间（https://www.luogu.com.cn/problem/P3865）使用ST表预处理区间的gcd
P1816 忠诚（https://www.luogu.com.cn/problem/P1816）使用ST表预处理区间的最小值
P2412 查单词（https://www.luogu.com.cn/problem/P2412）预处理字典序之后使用ST表查询静态区间最大字典序
P2880 [USACO07JAN] Balanced Lineup G（https://www.luogu.com.cn/problem/P2880）查询区间最大值与最小值
P5097 [USACO04OPEN]Cave Cows 2（https://www.luogu.com.cn/problem/P5097）静态区间最小值
P5648 Mivik的神力（https://www.luogu.com.cn/problem/P5648）使用倍增 ST 表查询区间最大值的索引，使用单调栈建树计算距离

================================CodeForces================================
D. Max GEQ Sum（https://codeforces.com/problemset/problem/1691/D）单调栈枚举加ST表最大值最小值查询
D. Friends and Subsequences（https://codeforces.com/problemset/problem/689/D）根据单调性使用二分加ST表进行个数计算
D. Yet Another Yet Another Task（https://codeforces.com/problemset/problem/1359/D）单调栈枚举加ST表最大值最小值查询
B. Integers Have Friends（https://codeforces.com/problemset/problem/1548/B）ST表查询区间gcd并枚举数组开头，二分确定长度
F. Ant colony（https://codeforces.com/problemset/problem/474/F）稀疏表计算最小值和gcd，并使用二分查找计数
E. MEX of LCM（https://codeforces.com/contest/1834/problem/E）经典计算连续子数组的lcm信息
E. Iva & Pav（https://codeforces.com/contest/1878/problem/E）经典计算连续子数组的and信息

================================AcWing====================================
109. 天才ACM（https://www.acwing.com/problem/content/111/）贪心加倍增计算最少分段数

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
        # 相当于一条链的树倍增求LCA
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
            return math.lcm(a, b)
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


class SparseTable2D:
    def __init__(self, matrix, method="max"):
        m, n = len(matrix), len(matrix[0])
        a, b = int(math.log2(m)) + 1, int(math.log2(n)) + 1

        if method == "max":
            self.fun = self.max
        elif method == "min":
            self.fun = self.min
        elif method == "gcd":
            self.fun = self.gcd
        elif method == "lcm":
            self.fun = self.min
        elif method == "or":
            self.fun = self._or
        else:
            self.fun = self._and

        self.dp = [[[[0 for _ in range(b)] for _ in range(a)] for _ in range(1000)] for _ in range(1000)]

        for i in range(a):
            for j in range(b):
                for x in range(m - (1 << i) + 1):
                    for y in range(n - (1 << j) + 1):
                        if i == 0 and j == 0:
                            self.dp[x][y][i][j] = matrix[x][y]
                        elif i == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i][j - 1]])
                        elif j == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j]])
                        else:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y + (1 << (j - 1))][i - 1][j - 1]])
        return

    @staticmethod
    def max(args):
        return reduce(max, args)

    @staticmethod
    def min(args):
        return reduce(min, args)

    @staticmethod
    def gcd(args):
        return reduce(gcd, args)

    @staticmethod
    def lcm(args):
        return reduce(lcm, args)

    @staticmethod
    def _or(args):
        return reduce(or_, args)

    @staticmethod
    def _and(args):
        return reduce(and_, args)

    def query(self, x, y, x1, y1):
        # 索引从0开始
        k = int(math.log2(x1 - x + 1))
        p = int(math.log2(y1 - y + 1))
        ans = self.fun([self.dp[x][y][k][p],
                        self.dp[x1 - (1 << k) + 1][y][k][p],
                        self.dp[x][y1 - (1 << p) + 1][k][p],
                        self.dp[x1 - (1 << k) + 1][y1 - (1 << p) + 1][k][p]])
        return ans


class SparseTableIndex:
    def __init__(self, lst, fun="max"):
        # 只要fun满足单调性就可以进行静态区间查询极值所在的索引
        self.fun = fun
        self.n = len(lst)
        self.lst = lst
        self.f = [[0] * (int(math.log2(self.n)) + 1)
                  for _ in range(self.n + 1)]
        self.gen_sparse_table()
        return

    def gen_sparse_table(self):
        # 相当于一条链的树倍增求LCA
        for i in range(1, self.n + 1):
            self.f[i][0] = i - 1
        for j in range(1, int(math.log2(self.n)) + 1):
            for i in range(1, self.n - (1 << j) + 2):
                a = self.f[i][j - 1]
                b = self.f[i + (1 << (j - 1))][j - 1]
                if self.fun == "max":
                    self.f[i][j] = a if self.lst[a] > self.lst[b] else b
                elif self.fun == "min":
                    self.f[i][j] = a if self.lst[a] < self.lst[b] else b
        return

    def query(self, left, right):
        # 查询数组的索引 left 和 right 从 1 开始
        k = int(math.log2(right - left + 1))
        a = self.f[left][k]
        b = self.f[right - (1 << k) + 1][k]
        if self.fun == "max":
            return a if self.lst[a] > self.lst[b] else b
        elif self.fun == "min":
            return a if self.lst[a] < self.lst[b] else b


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

    @staticmethod
    def ac_109(ac=FastIO()):

        def merge(lst1, lst2):
            a, b = len(lst1), len(lst2)
            x = y = 0
            res = []
            while x < a or y < b:
                if x == a or (y < b and lst2[y] < lst1[x]):
                    res.append(lst2[y])
                    y += 1
                else:
                    res.append(lst1[x])
                    x += 1
            return res

        def check(lst1):
            k = len(lst1)
            x, y = 0, k - 1
            res = cnt = 0
            while x < y and cnt < m:
                res += (lst1[x] - lst1[y]) ** 2
                if res > t:
                    return False
                x += 1
                y -= 1
                cnt += 1
            return True

        # 模板：利用倍增与归并排序的思想进行数组划分
        for _ in range(ac.read_int()):
            n, m, t = ac.read_ints()
            nums = ac.read_list_ints()
            ans = i = 0
            while i < n:
                p = 1
                lst = [nums[i]]
                right = i
                while p and right < n:
                    cur = nums[right + 1:right + p + 1]
                    cur.sort()
                    tmp = merge(lst, cur)
                    if check(tmp):
                        lst = tmp[:]
                        right += p
                        p *= 2
                    else:
                        p //= 2
                ans += 1
                i = right + 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p5648(ac=FastIO()):
        # 模板：使用倍增 ST 表查询区间最大值的索引，使用单调栈建树计算距离
        n, t = ac.read_ints()
        nums = ac.read_list_ints()
        post = [n]*n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                post[stack.pop()] = i
            stack.append(i)
        edge = [[] for _ in range(n + 1)]
        for i in range(n):
            edge[post[i]].append(i)
        # 建树计算距离
        sub = [0] * (n + 1)
        stack = [n]
        while stack:
            i = stack.pop()
            for j in edge[i]:
                sub[j] = sub[i] + nums[j] * (i - j)
                stack.append(j)
        # 区间最大值索引
        st = SparseTableIndex(nums)
        last_ans = 0
        for _ in range(t):
            u, v = ac.read_ints()
            left = 1 + (u ^ last_ans) % n
            q = 1 + (v ^ (last_ans + 1)) % (n - left + 1)
            right = left + q - 1
            ceil_ind = st.query(left, right)
            last_ans = sub[left - 1] - sub[ceil_ind] + nums[ceil_ind] * (right - ceil_ind)
            ac.st(last_ans)
        return

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        # 模板：最大公因数等于 K 的子数组数目
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.gcd(p, num)
                if x % k == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if num % k == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2470(nums: List[int], k: int) -> int:
        # 模板：最小公倍数为 K 的子数组数目
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.lcm(p, num)
                if k % x == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if k % num == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2411(nums: List[int]) -> List[int]:
        # 模板：经典计算最大或值的最短连续子数组
        n = len(nums)
        ans = [0] * n
        post = dict()
        for i in range(n - 1, -1, -1):
            cur = dict()
            num = nums[i]
            for x in post:
                y = cur.get(x | num, inf)
                cur[x | num] = y if y < post[x] else post[x]
            cur[num] = i
            post = cur
            ans[i] = post[max(post)] - i + 1
        return ans

    @staticmethod
    def cf_1878e(ac=FastIO()):
        # 解法：经典计算连续子数组的and信息
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            q = ac.read_int()
            query = [dict() for _ in range(n)]
            res = []
            for _ in range(q):
                ll, k = ac.read_ints()
                ll -= 1
                res.append([ll, k])
                query[ll][k] = -2
            post = dict()
            for i in range(n - 1, -1, -1):
                cur = dict()
                num = nums[i]
                for p in post:
                    x = p & num
                    if x not in cur or post[p] > cur[x]:
                        cur[x] = post[p]
                if num not in cur:
                    cur[num] = i
                lst = sorted(query[i].keys(), reverse=True)
                val = [[num, cur[num]] for num in cur]
                val.sort(reverse=True)
                right = -2
                m = len(val)
                p = 0
                for ke in lst:
                    while p < m and val[p][0] >= ke:
                        _, xx = val[p]
                        if xx > right:
                            right = xx
                        p += 1
                    query[i][ke] = right
                post = cur.copy()
            ac.lst([query[ll][k] + 1 for ll, k in res])
        return

    @staticmethod
    def lc_1521(arr: List[int], target: int) -> int:
        # 模板：经典计算与目标值最接近的连续子数组位运算与值
        ans = abs(arr[0] - target)
        pre = {arr[0]}
        for num in arr[1:]:
            pre = {num & p for p in pre}
            pre.add(num)
            for x in pre:
                if abs(x - target) < ans:
                    ans = abs(x - target)
        return ans


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

    def test_sparse_table_2d_max_min(self):

        # 二维稀疏表
        m = n = 50
        high = 100000
        grid = [[random.randint(0, high) for _ in range(n)] for _ in range(m)]

        for method in ["max", "min", "lcm", "gcd", "or", "and"]:
            st = SparseTable2D(grid, method)
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)

            ans1 = st.query(x1, y1, x2, y2)
            ans2 = st.fun([st.fun(g[y1:y2 + 1]) for g in grid[x1:x2 + 1]])
            assert ans1 == ans2
        return


if __name__ == '__main__':
    unittest.main()
