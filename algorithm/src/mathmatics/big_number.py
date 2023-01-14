"""

"""
"""
算法：大数分解、素数判断、高精度计算
功能：xxx
题目：
Lxxxx xxxx（https://leetcode.cn/problems/shortest-palindrome/）xxxx
P1920 成功密码（https://www.luogu.com.cn/problem/P1920）预估高精度计算与公式 -ln(1-x) = sum(x**i/i for in range(1, n+1)) 其中 n 趋近于无穷
P1729 计算e（https://www.luogu.com.cn/problem/P1729）高精度计算e小数位
P1727 计算π（https://www.luogu.com.cn/problem/P1727）高精度计算π小数位
P1530 [USACO2.4]分数化小数 Fractions to Decimals（https://www.luogu.com.cn/problem/P1530）分数化为小数
P1517 高精求小数幂（https://www.luogu.com.cn/record/list?user=739032&status=12&page=5）高精度计算小数的幂值
P2388 阶乘之乘（https://www.luogu.com.cn/problem/P2388）阶乘之乘后缀0的个数
172. 阶乘后的零（https://leetcode.cn/problems/factorial-trailing-zeroes/）阶乘后缀0的个数

参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from decimal import Decimal, getcontext, MAX_PREC
getcontext().prec = MAX_PREC


class HighPrecision:
    def __init__(self):
        return

    @staticmethod
    def factorial_to_factorial(n):
        # 模板：计算1!*2!***n!的后缀0个数
        ans = 0
        num = 5
        while num <= n:
            ans += num * (n // num) * (n // num - 1) // 2
            ans += (n // num) * (n % num + 1)
            num *= 5
        return ans

    @staticmethod
    def factorial_to_zero(n):
        # 模板：计算n!的后缀0个数
        ans = 0
        while n:
            ans += n // 5
            n //= 5
        return ans

    @staticmethod
    def float_pow(r, n):
        # 高精度计算小数的幂值
        ans = (Decimal(r) ** int(n)).normalize()
        ans = "{:f}".format(ans)
        return ans

    @staticmethod
    def fraction_to_decimal(numerator: int, denominator: int) -> str:
        # 分数转换为有理数或者无限循环小数
        if numerator % denominator == 0:
            return str(numerator // denominator) + ".0"
        ans = []
        if numerator * denominator < 0:
            ans.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)

        ans.append(str(numerator // denominator))
        numerator %= denominator
        ans.append(".")
        reminder = numerator % denominator
        dct = dict()
        while reminder and reminder not in dct:
            dct[reminder] = len(ans)
            reminder *= 10
            ans.append(str(reminder // denominator))
            reminder %= denominator
        if reminder in dct:
            ans.insert(dct[reminder], "(")
            ans.append(")")
        return "".join(ans)

    @staticmethod
    def decimal_to_fraction(st):

        def sum_fraction(tmp):
            # 分数相加
            mu = tmp[0][1]
            for ls in tmp[1:]:
                mu = math.lcm(mu, ls[1])
            zi = sum(ls[0] * mu // ls[1] for ls in tmp)
            mz = math.gcd(mu, zi)
            return [zi // mz, mu // mz]

        # 有理数或无限循环小数转换为分数
        if "." in st:
            lst = st.split(".")
            integer = [int(lst[0]), 1] if lst[0] else [0, 1]
            if "(" not in lst[1]:
                non_repeat = [int(lst[1]), 10 ** len(lst[1])
                              ] if lst[1] else [0, 1]
                repeat = [0, 1]
            else:
                pre, post = lst[1].split("(")
                non_repeat = [int(pre), 10 ** len(pre)] if pre else [0, 1]
                post = post[:-1]
                repeat = [int(post), int("9" * len(post)) * 10 ** len(pre)]
        else:

            integer = [int(st), 1]
            non_repeat = [0, 1]
            repeat = [0, 1]
        return sum_fraction([integer, non_repeat, repeat])


class TestGeneral(unittest.TestCase):

    def test_high_percision(self):
        hp = HighPrecision()
        assert hp.float_pow("98.999", "5") == "9509420210.697891990494999"

        assert hp.fraction_to_decimal(45, 56) == "0.803(571428)"
        assert hp.fraction_to_decimal(2, 1) == "2.0"
        assert hp.decimal_to_fraction("0.803(571428)") == [45, 56]
        assert hp.decimal_to_fraction("2.0") == [2, 1]
        return


if __name__ == '__main__':
    unittest.main()
