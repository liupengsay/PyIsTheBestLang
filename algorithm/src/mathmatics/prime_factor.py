
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
from algorithm.src.fast_io import FastIO


"""
算法：质因数分解、因数分解、素数筛、线性筛、欧拉函数、pollard_rho、Meissel–Lehmer 算法（计算范围内素数个数）
功能：素数有关计算
题目：


===================================力扣===================================
264. 丑数 II（https://leetcode.cn/problems/ugly-number-ii/）只含2、3、5质因数的第 n 个丑数
1201. 丑数 III（https://leetcode.cn/problems/ugly-number-iii/）只含特定因子数即能被其中一个数整除的第 n 个丑数
313. 超级丑数（https://leetcode.cn/problems/super-ugly-number/）只含某些特定质因数的第 n 个丑数
6364. 无平方子集计数（https://leetcode.cn/problems/count-the-number-of-square-free-subsets/）非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）
1994. 好子集的数目（https://leetcode.cn/problems/the-number-of-good-subsets/）非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）

===================================洛谷===================================
P1865 A % B Problem（https://www.luogu.com.cn/problem/P1865）通过线性筛素数后进行二分查询区间素数个数
P1748 H数（https://www.luogu.com.cn/problem/P1748）丑数可以使用堆模拟可以使用指针递增也可以使用容斥原理与二分进行计算
P2723 [USACO3.1]丑数 Humble Numbers（https://www.luogu.com.cn/problem/P2723）第n小的只含给定素因子的丑数
P1592 互质（https://www.luogu.com.cn/problem/P1592）使用二分与容斥原理计算与 n 互质的第 k 个正整数
P2926 [USACO08DEC]Patting Heads S（https://www.luogu.com.cn/problem/P2926）素数筛或者因数分解计数统计可被数列其他数整除的个数
P5535 【XR-3】小道消息（https://www.luogu.com.cn/problem/P5535）素数is_prime5判断加贪心脑筋急转弯
P1876 开灯（https://www.luogu.com.cn/problem/P1876）经典好题，理解完全平方数的因子个数为奇数，其余为偶数
P7588 双重素数（2021 CoE-II A）（https://www.luogu.com.cn/problem/P7588）素数枚举计算，优先使用is_prime4
P7696 [COCI2009-2010#4] IKS（https://www.luogu.com.cn/problem/P7696）数组，每个数进行质因数分解，然后均匀分配质因子
P4718 【模板】Pollard's rho 算法（https://www.luogu.com.cn/problem/P4718）使用pollard_rho进行质因数分解与素数判断

==================================AtCoder=================================

================================CodeForces================================
C. Hossam and Trainees（https://codeforces.com/problemset/problem/1771/C）使用pollard_rho进行质因数分解
D. Two Divisors（https://codeforces.com/problemset/problem/1366/D）计算最小的质因子，使用构造判断是否符合条件

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_6334(nums: List[int]) -> int:
        # 模板：非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）
        dct = {2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30}
        # 集合为质数因子幂次均为 1
        mod = 10 ** 9 + 7
        cnt = Counter(nums)
        pre = defaultdict(int)
        for num in cnt:
            if num in dct:
                cur = pre.copy()
                for p in pre:
                    if math.gcd(p, num) == 1:
                        cur[p * num] += pre[p] * cnt[num]
                        cur[p * num] %= mod
                cur[num] += cnt[num]
                pre = cur.copy()
        # 1 需要特殊处理
        p = pow(2, cnt[1], mod)
        ans = sum(pre.values()) * p
        ans += p - 1
        return ans % mod

    @staticmethod
    def cf_1366d(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        ceil = max(nums)

        # 模板：利用线性筛的思想计算最小的质因数
        min_div = [i for i in range(ceil + 1)]
        for i in range(2, len(min_div)):
            if min_div[i] != i:
                continue
            if i * i >= len(min_div):
                break
            for j in range(i, len(min_div)):
                if i * j >= len(min_div):
                    break
                if min_div[i * j] == i * j:
                    min_div[i * j] = i

        # 构造结果
        ans1 = []
        ans2 = []
        for num in nums:
            p = min_div[num]
            v = num
            while v % p == 0:
                v //= p
            if v == 1:
                # 只有一个质因子
                ans1.append(-1)
                ans2.append(-1)
            else:
                ans1.append(v)
                ans2.append(num // v)
        ac.lst(ans1)
        ac.lst(ans2)
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
