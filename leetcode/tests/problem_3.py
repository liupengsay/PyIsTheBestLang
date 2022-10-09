
import bisect
import itertools
import random
from typing import List
import heapq
import math
import re
import unittest
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations


def doubling(s):
    # sa[i]:排名为i的后缀的起始位置
    # rk[i]:起始位置为i的后缀的排名
    n = len(s)
    sa = []
    rk = []
    for i in range(n):
        rk.append(ord(s[i]) - ord('a'))  # 刚开始时，每个后缀的排名按照它们首字母的排序
        sa.append(i)  # 而排名第i的后缀就是从i开始的后缀

    l = 0  # l是已经排好序的长度，现在要按2l长度排序
    sig = 26  # sig是unique的排名的个数，初始是字符集的大小
    while True:
        p = []
        # 对于长度小于l的后缀来说，它们的第二关键字排名肯定是最小的，因为都是空的
        for i in range(n - l, n):
            p.append(i)
        # 对于其它长度的后缀来说，起始位置在`sa[i]`的后缀排名第i，而它的前l个字符恰好是起始位置为`sa[i]-l`的后缀的第二关键字
        for i in range(n):
            if sa[i] >= l:
                p.append(sa[i] - l)
        # 然后开始基数排序，先对第一关键字进行统计
        # 先统计每个值都有多少
        cnt = [0] * sig
        for i in range(n):
            cnt[rk[i]] += 1
        # 做个前缀和，方便基数排序
        for i in range(1, sig):
            cnt[i] += cnt[i - 1]
        # 然后利用基数排序计算新sa
        for i in range(n - 1, -1, -1):
            cnt[rk[p[i]]] -= 1
            sa[cnt[rk[p[i]]]] = p[i]
        # 然后利用新sa计算新rk

        def equal(i, j, l):
            if rk[i] != rk[j]:
                return False
            if i + l >= n and j + l >= n:
                return True
            if i + l < n and j + l < n:
                return rk[i + l] == rk[j + l]
            return False
        sig = -1
        tmp = [None] * n
        for i in range(n):
            # 直接通过判断第一关键字的排名和第二关键字的排名来确定它们的前2l个字符是否相同
            if i == 0 or not equal(sa[i], sa[i - 1], l):
                sig += 1
            tmp[sa[i]] = sig
        rk = tmp
        sig += 1
        if sig == n:
            break
        # 更新有效长度
        l = l << 1 if l > 0 else 1
    # 计算height数组
    k = 0
    height = [0] * n
    for i in range(n):
        if rk[i] > 0:
            j = sa[rk[i] - 1]
            while i + k < n and j + k < n and s[i + k] == s[j + k]:
                k += 1
            height[rk[i]] = k
            k = max(0, k - 1)  # 下一个height的值至少从max(0,k-1)开始
    return sa, rk, height


class Solution:
    def robotWithString(self, s: str) -> str:
        n = len(s)

        # 动态规划记录右侧的最小字母
        right = ["z"] * (n + 1)
        for i in range(n - 1, -1, -1):
            right[i] = min(right[i + 1], s[i])

        stack = []
        ans = ""
        for i in range(n):
            stack.append(s[i])
            # 假如当前栈顶为最小字母则出栈加到最终结果
            while stack and stack[-1] <= right[i + 1]:
                ans += stack.pop()
        return ans + "".join(stack[::-1])


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxHappyGroups(batchSize = 3, groups = [1,2,3,4,5,6]) == 4
        assert Solution().maxHappyGroups(batchSize = 4, groups = [1,3,2,5,2,2,1,6]) == 4
        assert Solution().maxHappyGroups(3, [844438225,657615828,355556135,491931377,644089602,30037905,863899906,246536524,682224520]) == 6
        assert Solution().maxHappyGroups(8, [244197059,419273145,329407130,44079526,351372795,200588773,340091770,851189293,909604028,621703634,959388577,989293607,325139045,263977422,358987768,108391681,584357588,656476891,621680874,867119215,639909909,98831415,263171984,236390093,21876446]) == 13
        return


if __name__ == '__main__':
    unittest.main()
