"""
算法：LGV引理
功能：用来处理有向无环图上不相交路径计数问题
题目：P6657 【模板】LGV 引理（https://www.luogu.com.cn/problem/P6657）
参考：OI WiKi（https://oi-wiki.org/graph/lgv/）
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


class LGV:
    def __init__(self):
        return

    @staticmethod
    def get_result(a, b, c, d):
        """从[a,b]走到[c,d]的方案数"""
        return math.comb(d+c-a-b, c-a)

    def compute(self, start, end, n):
        """从[start_i,1]到[end_j,n]的走法"""
        m = len(start)
        grid = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if end[j] >= start[i]:
                    grid[i][j] = self.get_result(start[i], 1, end[j], n)
        ans = np.linalg.det(np.array(grid))
        return int(np.around(ans))


    def gao_si(self, m1):
        """
        :param m1: 待求值的n阶行列式
        :return: 行列式的值
        """
        x = len(m1)
        for po in range(x - 1):
            if m1[po, po]:
                rem = m1[po, po]

                # 行操作
                for c in range(po, x):
                    m1[po, c] = m1[po, c] / rem

                # 列操作
                for m in range(po + 1, x):  # 固定行
                    rem1 = m1[m, po]
                    for n in range(po, x):
                        m1[m, n] -= rem1 * m1[po, n]

            else:
                for cur in range(po + 1, x):
                    if m1[cur, po]:
                        tmp = m1[cur]
                        tmp1 = tmp.copy()
                        m1[cur] = m1[po]
                        m1[po] = tmp1
                        return self.gao_si(m1)
                    else:
                        if cur == x - 1:
                            return 0.0
        num = 1
        for i in range(x):
            num *= m1[i, i]
        return num

    def compute2(self, start, end, n):
        """从[start_i,1]到[end_j,n]的走法"""
        m = len(start)
        grid = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if end[j] >= start[i]:
                    grid[i][j] = self.get_result(start[i], 1, end[j], n)

        #assert self.gao_si(np.array(grid)) == self.get_det(np.array(grid))
        ans = self.gao_si(np.array(grid))
        return int(np.around(ans))

    # 自己编写算法求解
    def get_det(self, a):
        mutifier=1
        i_value,j_value=a.shape
        # 如果第一行第一列不为0
        if a[0][0]!=0:
            for col in range(j_value):
                for i in range(i_value):
                    if i>=col+1:
                    # 需要消元的列
                        # 如果不等于0,消元
                        if a[i][col]!=0:
                            k=-1*a[i][col]/a[0+col][col]
                            for j in range(col,j_value):
                                a[i][j]=a[i][j]+k*a[0+col][j]
            value=mutifier
            for i in range(i_value):
                for j in range(j_value):
                    if i==j:
                        value*=a[i][j]
            return value
        if a[0][0]==0:
            col=0
            for j in range(j_value):
                if a[0][j]!=0:
                    if col==0:
                        col=j
            # 如果一行为0，值为0
            if col==0:
                return 0
            # 如若不是，就交换两列
            else:
                first_result=[]
                second_result=[]
                for i in range(i_value):
                    first_result.append(a[i][0])
                    second_result.append(a[i][col])
                a[:,0]=second_result
                a[:,col]=first_result
                return -1*self.get_det(a)


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        lgv = LGV()
        assert lgv.get_result(1, 1, 2, 2) == 2
        return


if __name__ == '__main__':
    unittest.main()
