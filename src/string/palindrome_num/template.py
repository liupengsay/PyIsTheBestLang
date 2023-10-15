import unittest
import bisect
from collections import defaultdict

from src.mathmatics.number_theory import NumberTheory




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



