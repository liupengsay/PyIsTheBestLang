import math
import unittest

from src.fast_io import FastIO
from src.mathmatics.lexico_graphical_order import LexicoGraphicalOrder

"""
算法：康托展开
功能：康托展开可以用来求一个 1~n 的任意排列的排名，以及任意排名的 1~n 排列
题目：

===================================洛谷===================================
P3014 [USACO11FEB]Cow Line S（https://www.luogu.com.cn/problem/P3014）计算全排列的排名与排名对应的全排列
P5367 【模板】康托展开（https://www.luogu.com.cn/problem/P5367）计算排列的排名



===================================AcWing===================================
5052. 排列（https://www.acwing.com/problem/content/5055/）经典康托展开与BFS搜索，根据排列数确定最多末尾重排的长度


参考：OI WiKi（https://oi-wiki.org/math/combinatorics/cantor/）
"""


class CantorExpands:
    def __init__(self, n, mod=10**9 + 7):
        self.mod = mod
        self.dp = [1] * (n + 1)
        for i in range(2, n):
            self.dp[i] = i * self.dp[i - 1] % mod
        return

    def array_to_rank(self, nums):
        lens = len(nums)
        out = 1
        for i in range(lens):
            res = 0
            fact = self.dp[lens - i - 1]
            for j in range(i + 1, lens):
                if nums[j] < nums[i]:
                    res += 1
            out += res * fact
            out %= self.mod
        return out

    def rank_to_array(self, n, k):
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = self.dp[len(nums) - 1]
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p5367(ac=FastIO()):
        # 模板：计算数组在 1 到 n 的全排列当中的排名
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353
        ce = CantorExpands(n, mod)
        ac.st(ce.array_to_rank(nums) % mod)
        return

    @staticmethod
    def lg_p3014_1(ac=FastIO()):
        # 模板：康托展开也可以使用字典序贪心计算
        n, q = ac.read_ints()
        ct = CantorExpands(n, mod=math.factorial(n + 2))
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(ct.rank_to_array(n, lst[0]))
            else:
                ac.st(ct.array_to_rank(lst))
        return

    @staticmethod
    def lg_p3014_2(ac=FastIO()):
        # 模板：康托展开也可以使用字典序贪心计算
        n, q = ac.read_ints()
        og = LexicoGraphicalOrder()
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(og.get_kth_subset_perm(n, lst[0]))
            else:
                ac.st(og.get_subset_perm_kth(n, lst))
        return

    @staticmethod
    def ac_5052(ac=FastIO()):
        # 模板：经典康托展开与BFS搜索，根据排列数确定最多末尾重排的长度
        n, k = ac.read_ints()
        low = ac.max(1, n-12)
        high = n

        if math.factorial(high-low+1) < k:
            ac.st(-1)
            return

        lst = list(range(low, high+1))
        m = high-low+1
        ce = CantorExpands(m, 10**20)
        ind = ce.rank_to_array(m, k)
        perm = [lst[i-1] for i in ind]
        ans = 0
        stack = [0]
        while stack:
            x = stack.pop()
            if x > n:
                continue
            stack.append(x*10+4)
            stack.append(x*10+7)
            if 1 <= x <= n:
                a_x = x if x < low else perm[x-low]
                if all(w in "47" for w in str(a_x)):
                    ans += 1
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_cantor_expands(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
