"""

"""
"""
算法：栈
功能：模拟题中常见，如括号之类的，后进先出
题目：
L2197 替换数组中的非互质数（https://leetcode.cn/problems/replace-non-coprime-numbers-in-array/）结合数学使用栈进行模拟
P1944 最长括号匹配（https://www.luogu.com.cn/problem/P1944）最长连续合法括号字串长度
394. 字符串解码（https://leetcode.cn/problems/decode-string/）经典解码带括号成倍的字符和数字
P2201 数列编辑器（https://www.luogu.com.cn/problem/P2201）双栈模拟指针移动同时记录前缀和与前序最大前缀和
P4387 【深基15.习9】验证栈序列（https://www.luogu.com.cn/problem/P4387）模拟入栈出栈队列判断是否可行

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


class ClassName:
    def __init__(self):
        return

    def gen_result(self):
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
