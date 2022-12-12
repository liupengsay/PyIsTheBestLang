"""
算法：位运算相关技巧
功能：进行二进制上的位操作
题目：xx（xx）
参考：OI WiKi（xx）
https://blog.csdn.net/qq_35473473/article/details/106320878
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
            if graycode[i] == binary[i-1]:
                b = 0
            else:
                b = 1
            binary.append(str(b))
        return int("0b"+ ''.join(binary), 2)
    
    @staticmethod
    def integer_to_graycode(integer):
        # 二进制转格雷码
        binary = bin(integer).replace('0b', '')
        graycode = list()
        binay_len = len(binary)
        graycode.append(binary[0])
        for i in range(1, binay_len):
            if binary[i-1] == binary[i]:
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
