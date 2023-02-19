
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

"""

算法：位运算相关技巧
功能：进行二进制上的位操作，包括与、异或、或、取反，通常使用按位思考与举例的方式寻找规律
题目：

===================================力扣===================================
2354. 优质数对的数目（https://leetcode.cn/problems/number-of-excellent-pairs/）需要脑筋急转弯确定位 1 的规律进行哈希计数枚举即可
260. 只出现一次的数字 III（https://leetcode.cn/problems/single-number-iii/）利用位运算两个相同元素异或和为0的特点，以及lowbit进行分组确定两个只出现一次的元素
6365. 将整数减少到零需要的最少操作数（https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/）n 加上或减去 2 的某个幂使得 n 变为 0 的最少操作数

===================================洛谷===================================
P5657 格雷码（https://www.luogu.com.cn/problem/P5657）计算编号为 k 的二进制符，并补前缀 0 为 n 位
P6102 [EER2]谔运算（https://www.luogu.com.cn/problem/P6102）经典位运算加和题目，按位计算，按照位0与位1的数量进行讨论
P7442 「EZEC-7」维护序列（https://www.luogu.com.cn/problem/P7442）观察操作规律，使用位运算模拟操作
P7617 [COCI2011-2012#2] KOMPIĆI（https://www.luogu.com.cn/problem/P7617）使用位运算枚举
P7627 [COCI2011-2012#1] X3（https://www.luogu.com.cn/problem/P7627）经典按位操作枚举计算个数
P7649 [BalticOI 2004 Day 1] SCALES（https://www.luogu.com.cn/problem/P7649）三进制计算，贪心模拟砝码放置

================================CodeForces================================
https://codeforces.com/problemset/problem/305/C（利用二进制加减的思想进行解题）
https://codeforces.com/problemset/problem/878/A（位运算的操作理解）
http://codeforces.com/problemset/problem/282/C（利用位运算的特性进行判断）


参考：OI WiKi（xx）
https://blog.csdn.net/qq_35473473/article/details/106320878
"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6365(num):
        # 模板：n 加上或减去 2 的某个幂使得 n 变为 0 的最少操作数
        @lru_cache(None)
        def dfs(n):
            if not n:
                return 0
            if bin(n).count("1") == 1:
                return 1
            lowbit = n & (-n)
            return 1 + min(dfs(n - lowbit), dfs(n + lowbit))

        def greedy(n):
            # 对应有 O(logn) 贪心解法
            s = bin(n)[2:][::-1]
            ans = cnt = 0
            m = len(s)
            for i in range(m):
                if s[i] == "1":
                    cnt += 1
                else:
                    if cnt == 1:
                        ans += 1
                        cnt = 0
                    elif cnt >= 2:
                        if i + 1 < m and s[i + 1] == "1":
                            ans += 1
                            cnt = 1
                        else:
                            ans += 2
                            cnt = 0
            if cnt:
                ans += 1 if cnt == 1 else 2
            return ans

        # 更优解法 bin(n ^ (3 * n)).count("1")
        return dfs(num)
    @staticmethod
    def lc_260(nums):
        # 模板：找出数组当中两个只出现一次的数（其余数保证出现两次）
        s = reduce(xor, nums)
        low = s & (-s)
        ans = [0, 0]
        for num in nums:
            if num & low:
                ans[0] ^= num
            else:
                ans[1] ^= num
        return ans


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def graycode_to_integer(graycode):
        # 格雷码转二进制
        graycode_len = len(graycode)
        binary = list()
        binary.append(graycode[0])
        for i in range(1, graycode_len):
            if graycode[i] == binary[i - 1]:
                b = 0
            else:
                b = 1
            binary.append(str(b))
        return int("0b" + ''.join(binary), 2)

    @staticmethod
    def integer_to_graycode(integer):
        # 二进制转格雷码
        binary = bin(integer).replace('0b', '')
        graycode = list()
        binay_len = len(binary)
        graycode.append(binary[0])
        for i in range(1, binay_len):
            if binary[i - 1] == binary[i]:
                g = 0
            else:
                g = 1
            graycode.append(str(g))
        return ''.join(graycode)

    @staticmethod
    def get_graycode(n):
        # n位数格雷码
        code = [0, 1]
        for i in range(1, n):
            code.extend([(1 << i) + num for num in code[::-1]])
        return code


class TestGeneral(unittest.TestCase):

    def test_bit_operation(self):
        bo = BitOperation()

        lst = [bo.integer_to_graycode(i) for i in range(11)]
        print(lst)

        assert bo.integer_to_graycode(0) == "0"
        assert bo.integer_to_graycode(22) == "11101"
        assert bo.graycode_to_integer("10110") == 27

        n = 8
        code = bo.get_graycode(n)
        m = len(code)
        for i in range(m):
            assert bo.graycode_to_integer(bin(code[i])[2:]) == i
            assert bo.integer_to_graycode(i) == bin(code[i])[2:]
        return


if __name__ == '__main__':
    unittest.main()
