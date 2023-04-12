import random
import unittest
from typing import List

from sortedcontainers import SortedList
from algorithm.src.fast_io import FastIO

"""

算法：使用数组作为链表维护前驱后驱
功能：

题目：xx（xx）

===================================力扣===================================

===================================洛谷===================================


================================CodeForces================================
E. Two Teams（https://codeforces.com/contest/1154/problem/E）使用数组维护链表的前后节点信息

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1154e(ac=FastIO()):

        # 模板：使用链表维护前后的节点信息
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans = [0]*n
        pre = [i-1 for i in range(n)]
        nex = [i+1 for i in range(n)]
        ind = [0]*n
        for i in range(n):
            ind[nums[i]-1] = i

        step = 1
        for num in range(n-1, -1, -1):
            i = ind[num]
            if ans[i]:
                continue
            ans[i] = step
            left, right = pre[i], nex[i]
            for _ in range(k):
                if left != -1:
                    ans[left] = step
                    left = pre[left]
                else:
                    break
            for _ in range(k):
                if right != n:
                    ans[right] = step
                    right = nex[right]
                else:
                    break
            if left >= 0:
                nex[left] = right
            if right < n:
                pre[right] = left
            step = 3 - step
        ac.st("".join(str(x) for x in ans))
        return
