import math
import random
import unittest
from functools import reduce
from math import gcd
from operator import add
from itertools import accumulate
from typing import List

from src.fast_io import FastIO

"""
算法：BitSet、位集合、模拟区间状态变换、区间翻转
功能：通过相对移动，来减少计算复杂度，分为同向双指针，相反双指针，以及中心扩展法
参考：
题目：

===================================力扣===================================
2569. 更新数组后处理求和查询（https://leetcode.cn/problems/handling-sum-queries-after-update/）经典01线段树区间翻转与求和，也可以使用BitSet

===================================洛谷===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

================================CodeForces================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

================================AtCoder================================
D - FT Robot（https://atcoder.jp/contests/abc082/tasks/arc087_b）思维题，分开BFS平面坐标的x与y轴移动，使用bitset优化

================================AcWing================================
5037. 区间异或（https://www.acwing.com/problem/content/5040/）同CF242E，使用二十多个01线段树维护区间异或与区间加和

"""


class SegmentTreeBitSet:
    # 使用位运算模拟线段树进行区间01翻转操作
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # 索引从0开始翻转区间[b, c]
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        # 索引从0开始查询区间[b, c]的个数
        p = (1 << (c + 1)) - (1 << b)
        return (self.val & p).bit_count()


class Solution:
    def __int__(self):
        return

    @staticmethod
    def abc_82d(ac=FastIO()):
        # 模板：思维题，分开BFS平面坐标的x与y轴移动，使用bitset优化
        s = ac.read_str()
        x, y = ac.read_ints()
        ls = [len(t) for t in s.split("T")]
        pre_x = 1 << ls[0] + 8000
        for d in ls[2::2]:
            pre_x = (pre_x >> d) | (pre_x << d)

        pre_y = 1 << 8000
        for d in ls[1::2]:
            pre_y = (pre_y >> d) | (pre_y << d)
        ac.st("Yes" if pre_x & (1 << x + 8000) and pre_y & (1 << y + 8000) else "No")
        return

    @staticmethod
    def lc_2569_2(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # 模板：经典01线段树区间翻转与求和，也可以使用BitSet
        res = []
        seg = SegmentTreeBitSet()
        n = len(nums1)
        for i in range(n):
            if nums1[i]:
                seg.update(i, i)
        s = sum(nums2)
        for a, b, c in queries:
            if a == 1:
                seg.update(b, c)
            elif a == 2:
                s += seg.val.bit_count() * b
            else:
                res.append(s)
        return res

    @staticmethod
    def ac_5037_2(ac=FastIO()):
        # 模板：同CF242E，使用二十多个01线段树维护区间异或与区间加和
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegmentTreeBitSet() for _ in range(22)]
        for i in range(n):
            x = nums[i]
            for j in range(22):
                if x & (1 << j):
                    tree[j].update(i, i)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1 << j)*tree[j].query(ll, rr) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1 << j) & xx:
                        tree[j].update(ll, rr)
        return


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
