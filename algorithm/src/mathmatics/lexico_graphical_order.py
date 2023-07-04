import math
import random
import unittest
from itertools import permutations, combinations

from algorithm.src.fast_io import FastIO

"""
算法：字典序与字典序排名解析
功能：计算字典序第K小和某个对象的字典序rank、计算subset的字典序与解析、计算comb的字典序与解析、计算perm的字典序与解析
题目：

===================================力扣===================================
60. 排列序列（https://leetcode.cn/problems/permutation-sequence/）全排列的第 k 个排列
440. 字典序的第K小数字（https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/）经典面试题使用十叉树求解
1415. 长度为 n 的开心字符串中字典序第 k 小的字符串（https://leetcode.cn/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）类似思路经典字典序构造

===================================洛谷===================================
P1243 排序集合（https://www.luogu.com.cn/problem/P1243）求出第K小的子集
P1338 末日的传说（https://www.luogu.com.cn/problem/P1338）结合逆序对计数的字典序

P2524 Uim的情人节礼物·其之弐（https://www.luogu.com.cn/problem/P2524）计算全排列的字典序排名
P2525 Uim的情人节礼物·其之壱（https://www.luogu.com.cn/problem/P2525）计算全排列的上一个排列
P3014 [USACO11FEB]Cow Line S（https://www.luogu.com.cn/problem/P3014）计算全排列的排名与排名对应的全排列

================================CodeForces================================
B. K-th Beautiful String（https://codeforces.com/problemset/problem/1328/B）计算comb的字典序

参考：OI WiKi（xx）
"""


class LexicoGraphicalOrder:
    def __init__(self):
        return

    @staticmethod
    def get_kth_num(n, k):
        # 模板：求 1 到 n 范围内字典序第 k 小的数字
        def check():
            c = 0
            first = last = num
            while first <= n:
                c += min(last, n) - first + 1
                last = last * 10 + 9
                first *= 10
            return c

        # assert k <= n
        num = 1
        k -= 1
        while k:
            cnt = check()
            if k >= cnt:
                num += 1
                k -= cnt
            else:
                num *= 10
                k -= 1
        return num

    def get_num_kth(self, n, num):
        # 模板：求 1 到 n 范围内数字 num 的字典序
        x = str(num)
        low = 1
        high = n
        while low < high - 1:
            # 使用二分进行逆向工程
            mid = low + (high - low) // 2
            st = str(self.get_kth_num(n, mid))
            if st < x:
                low = mid
            elif st > x:
                high = mid
            else:
                return mid
        return low if str(self.get_kth_num(n, low)) == x else high

    @staticmethod
    def get_kth_subset(n, k):

        # 集合 [1,..,n] 的第 k 小的子集，总共有 1<<n 个子集
        # assert k <= (1 << n)
        ans = []
        if k == 1:
            # 空子集输出 0
            ans.append(0)
        k -= 1
        for i in range(1, n + 1):
            if k == 0:
                break
            if k <= pow(2, n - i):
                ans.append(i)
                k -= 1
            else:
                k -= pow(2, n - i)
        return ans

    def get_subset_kth(self, n, lst):

        # 集合 [1,..,n] 的子集 lst 的字典序
        low = 1
        high = n
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset(n, low) == lst else high

    @staticmethod
    def get_kth_subset_comb(n, m, k):
        # 集合 [1,..,n] 中选取 m 个元素的第 k 个 comb 选取排列
        # assert k <= math.comb(n, m)

        nums = list(range(1, n + 1))
        ans = []
        while k and nums and len(ans) < m:
            length = len(nums)
            c = math.comb(length - 1, m - len(ans) - 1)
            if c >= k:
                ans.append(nums.pop(0))
            else:
                k -= c
                nums.pop(0)
        return ans

    def get_subset_comb_kth(self, n, m, lst):
        # 集合 [1,..,n] 中选取 m 个元素的排列 lst 的字典序

        low = 1
        high = math.comb(n, m)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_comb(n, m, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_comb(n, m, low) == lst else high

    @staticmethod
    def get_kth_subset_perm(n, k):
        # 集合 [1,..,n] 中选取 n 个元素的第 k 个 perm 选取排列
        s = math.factorial(n)
        assert 1 <= k <= s
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = s//len(nums)
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
            s = single
        return ans

    def get_subset_perm_kth(self, n, lst):
        # 集合 [1,..,n] 中选取 n 个元素的 perm 全排列 lst 的字典序

        low = 1
        high = math.factorial(n)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_perm(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_perm(n, low) == lst else high


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1328b(ac=FastIO()):
        # 模板：计算comb选取的字典序
        for _ in range(ac.read_int()):
            n, k = ac.read_ints()
            ind = LexicoGraphicalOrder().get_kth_subset_comb(n, 2, n*(n-1)//2-k+1)
            ans = ["a"]*n
            for i in ind:
                ans[i-1] = "b"
            ac.st("".join(ans))
        return

    @staticmethod
    def lc_440(n, k):
        # 模板：计算 1 到 n 字典序第 k 小的数字
        return LexicoGraphicalOrder().get_kth_num(n, k)

    @staticmethod
    def lg_p1243(ac=FastIO()):
        # 模板：获取第 k 小的子集
        n, k = ac.read_ints()
        lst = LexicoGraphicalOrder().get_kth_subset(n, k)
        ac.lst(lst)
        return

    @staticmethod
    def lg_p2524(ac=FastIO()):
        # 模板：计算 1 到 n 的全排列中 lst 的字典序排名
        n = ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        rk = LexicoGraphicalOrder().get_subset_perm_kth(n, lst)
        ac.st(rk)
        return

    @staticmethod
    def lg_p3014(ac=FastIO()):

        # 模板：康托展开也可以使用字典序贪心计算
        n, q = ac.read_ints()
        og = LexicoGraphicalOrder()
        # ct = CantorExpands(n, mod=math.factorial(n + 2))
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(og.get_kth_subset_perm(n, lst[0]))
                # ac.lst(ct.rank_to_array(n, lst[0]))
            else:
                ac.st(og.get_subset_perm_kth(n, lst))
                # ac.st(ct.array_to_rank(lst))
        return

    @staticmethod
    def lc_60(n: int, k: int) -> str:
        #  模板：全排列的第 k 个排列
        ans = LexicoGraphicalOrder().get_kth_subset_perm(n, k)
        return "".join(str(x) for x in ans)


class TestGeneral(unittest.TestCase):

    def test_lexico_graphical_order(self):
        lgo = LexicoGraphicalOrder()

        n = 10**5
        nums = sorted([str(x) for x in range(1, n + 1)])
        for _ in range(100):
            i = random.randint(0, n - 1)
            num = nums[i]
            assert lgo.get_kth_num(n, i + 1) == int(num)
            assert lgo.get_num_kth(n, int(num)) == i + 1

        n = 10
        nums = []
        for i in range(1 << n):
            nums.append([j + 1 for j in range(n) if i & (1 << j)])
        nums.sort()
        nums[0] = [0]
        for _ in range(100):
            i = random.randint(0, n - 1)
            lst = nums[i]
            assert lgo.get_kth_subset(n, i + 1) == lst
            assert lgo.get_subset_kth(n, lst) == i + 1

        n = 10
        m = 4
        nums = []
        for item in combinations(list(range(1, n+1)), m):
            nums.append(list(item))
        for _ in range(100):
            i = random.randint(0, len(nums) - 1)
            lst = nums[i]
            assert lgo.get_kth_subset_comb(n, m, i+1) == lst
            assert lgo.get_subset_comb_kth(n, m, lst) == i + 1

        n = 8
        nums = []
        for item in permutations(list(range(1, n+1)), n):
            nums.append(list(item))
        for i, lst in enumerate(nums):
            lst = nums[i]
            assert lgo.get_kth_subset_perm(n, i + 1) == lst
            assert lgo.get_subset_perm_kth(n, lst) == i + 1
        return


if __name__ == '__main__':
    unittest.main()
