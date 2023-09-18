import unittest
import bisect
from collections import defaultdict

from src.mathmatics.number_theory import NumberTheory

"""
算法：回文数字枚举
功能：xxx
题目：

===================================力扣===================================
2081. k 镜像数字的和（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）枚举 10 进制回文数字并判断其 k 进制是否依然回文
866. 回文素数（https://leetcode.cn/problems/prime-palindrome/）枚举回文素数
564. 寻找最近的回文数（https://leetcode.cn/problems/find-the-closest-palindrome/）枚举字符的前半部分与后半部分
906. 超级回文数（https://leetcode.cn/problems/super-palindromes/）经典预处理所有的回文数其开方也是回文数
1088. 易混淆数 II（https://leetcode.cn/problems/confusing-number-ii/description/）模拟枚举所有的易混淆数

===================================洛谷===================================
P1609 最小回文数（https://www.luogu.com.cn/problem/P1609）枚举字符的前半部分与后半部分


参考：OI WiKi（xx）
"""


class PalindromeNum:
    def __init__(self):
        return

    @staticmethod
    def get_palindrome_num_1(n):
        # 模板：计算出长度小于等于 n 的所有正回文数
        dp = [[""], [str(i) for i in range(10)]]
        for k in range(2, n + 1):
            # 使用动态规划模拟对称的回文子串添加
            if k % 2 == 1:
                m = k // 2
                lst = []
                for st in dp[-1]:
                    for i in range(10):
                        lst.append(st[:m] + str(i) + st[m:])
                dp.append(lst)
            else:
                lst = []
                for st in dp[-2]:
                    for i in range(10):
                        lst.append(str(i) + st + str(i))
                dp.append(lst)

        # 取出所有的回文数字并排序
        nums = []
        for lst in dp:
            for num in lst:
                if num and num[0] != "0":
                    nums.append(int(num))
        nums.sort()
        # 计算出长度小于等于 n 的所有正回文数
        return nums

    @staticmethod
    def get_palindrome_num_2(n):
        assert n >= 1
        # 模板：计算出长度小于等于 n 的所有正回文数
        nums = list(range(1, 10))
        x = 1
        while len(str(x)) * 2 <= n:
            num = str(x) + str(x)[::-1]
            nums.append(int(num))
            if len(str(x)) * 2 + 1 <= n:
                for d in range(10):
                    nums.append(int(str(x) + str(d) + str(x)[::-1]))
            x += 1
        nums.sort()
        return nums

    @staticmethod
    def get_recent_palindrome_num(n: str) -> list:
        # 用原数的前半部分加一后的结果替换后半部分得到的回文整数。
        # 用原数的前半部分减一后的结果替换后半部分得到的回文整数。
        # 为防止位数变化导致构造的回文整数错误，因此直接构造 999…999 和 100…001 作为备选答案
        # 计算正整数 n 附近的回文数，获得最近的最小或者最大的回文数

        m = len(n)
        candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
        prefix = int(n[:(m + 1) // 2])
        for x in range(prefix - 1, prefix + 2):
            y = x if m % 2 == 0 else x // 10
            while y:
                x = x * 10 + y % 10
                y //= 10
            candidates.append(x)
        return candidates


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_906(left: str, right: str) -> int:
        # 模板：经典预处理所有的回文数其开方也是回文数
        nums = PalindromeNum().get_palindrome_num_2(10)
        res = [num * num for num in nums if str(num * num)[::-1] == str(num * num)]
        left = int(left)
        right = int(right)
        return bisect.bisect_right(res, right) - bisect.bisect_left(res, left)

    @staticmethod
    def lc_1088(n: int) -> int:
        # 模板：预处理后进行二分查找
        ind = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        pre = [0, 1, 6, 8, 9]
        res = pre[:]
        for _ in range(2, 10):
            nex = []
            for num in pre:
                for d in ind:
                    # 经典枚举方式
                    nex.append(num * 10 + d)
            res.extend(nex)
            pre = nex[:]
        res = sorted(set(res))

        def check(x):
            # check函数
            if x <= 0:
                return False
            s = str(x)
            t = int("".join(str(ind[int(w)]) for w in s[::-1]))
            return t != x

        res = [num for num in res if check(num)]
        res.append(1000000000)  # 注意边界
        return bisect.bisect_right(res, n)

    @staticmethod
    def lc_2081(k: int, n: int) -> int:
        # 模板：枚举 10 进制回文数字并判断其 k 进制是否依然回文
        dct = defaultdict(list)
        # 放到预处理
        nums = PalindromeNum().get_palindrome_num_2(12)
        for k in range(2, 10):
            for num in nums:
                lst = NumberTheory().get_k_bin_of_n(num, k)
                if lst == lst[::-1]:
                    dct[k].append(num)
                    if len(dct[k]) >= 30:
                        break
        return sum(dct[k][:n])


class TestGeneral(unittest.TestCase):

    def test_palindrome_num(self):
        pn = PalindromeNum()
        assert pn.get_palindrome_num_1(12) == pn.get_palindrome_num_2(12)

        n = "44"
        nums = pn.get_recent_palindrome_num(n)
        nums = [num for num in nums if num > int(n)]
        assert min(nums) == 55
        return


if __name__ == '__main__':
    unittest.main()
