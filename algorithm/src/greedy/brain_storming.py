"""

"""
"""
算法：贪心、逆向思维
功能：各种可证明不可证明的头脑风暴
题目：
L2499 让数组不相等的最小总代价（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）利用鸽巢原理贪心计算最小代价
L2449 使数组相似的最少操作次数（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）转换题意进行排序后用奇偶数贪心变换得到
L2448 使数组相等的最小开销（https://leetcode.cn/problems/minimum-cost-to-make-array-equal/）利用中位数的特点变换到带权重广义下中位数的位置是最优的贪心进行增减
L2412 完成所有交易的初始最少钱数（https://leetcode.cn/problems/minimum-money-required-before-transactions/）根据交易增长特点进行自定义排序
L2366 将数组排序的最少替换次数（https://leetcode.cn/problems/minimum-replacements-to-sort-the-array/）倒序贪心不断分解得到满足要求且尽可能大的值
L2350 不可能得到的最短骰子序列（https://leetcode.cn/problems/shortest-impossible-sequence-of-rolls/）脑筋急转弯本质上是求全排列出现的轮数
L2344 使数组可以被整除的最少删除次数（https://leetcode.cn/problems/minimum-deletions-to-make-array-divisible/）利用最大公约数贪心删除最少的元素
L2136 全部开花的最早一天（https://leetcode.cn/problems/earliest-possible-day-of-full-bloom/）贪心安排成长时间最长的先种
L2071 你可以安排的最多任务数目（https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/）使用贪心加二分进行极值判断
P1031 均分纸牌（https://www.luogu.com.cn/problem/P1031）贪心计算每个点的前缀和流量，需要补齐或者输出时进行计数
L0517 超级洗衣机（https://leetcode.cn/problems/super-washing-machines/）类似上题，计算最小的左右移动次数以及往左右的移动次数

P1684 考验（https://www.luogu.com.cn/problem/P1684）线性贪心满足条件即增加计数

P1658 购物（https://www.luogu.com.cn/problem/P1658）看似背包实则贪心
P2001 硬币的面值（https://www.luogu.com.cn/problem/P2001）看似背包实则贪心
P1620 漂亮字串（https://www.luogu.com.cn/problem/P1620）分类讨论进行贪心
P2255 [USACO14JAN]Recording the Moolympics S（https://www.luogu.com.cn/problem/P2255）两个指针进行贪心
P2327 [SCOI2005]扫雷（https://www.luogu.com.cn/problem/P2327）脑筋急转弯进行枚举

P2649 游戏预言（https://www.luogu.com.cn/problem/P2649）贪心，输的时候输最惨，赢的时候微弱优势

P1367 蚂蚁（https://www.luogu.com.cn/problem/P1367）脑筋急转弯，蚂蚁的相对移动位置排序还是不变
P1362 兔子数（https://www.luogu.com.cn/problem/P1362）找规律之后，进行广度优先搜索枚举
P1090 [NOIP2004 提高组] 合并果子 / [USACO06NOV] Fence Repair G（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）从小到大贪心合并
P1334 瑞瑞的木板（https://www.luogu.com.cn/problem/P1334）逆向思维的合并果子，从小到大合并
P1325 雷达安装（https://www.luogu.com.cn/problem/P1325）排序后进行贪心修建更新

P1250 种树（https://www.luogu.com.cn/problem/P1250）区间的贪心题，使用线段树修改区间与查询和，以及二分进行计算
P1230 智力大冲浪（https://www.luogu.com.cn/problem/P1230）排序后进行选取贪心
P1159 排行榜（https://www.luogu.com.cn/problem/P1159）使用队列贪心进行模拟
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
class BrainStorming:
    def __init__(self):
        return

    @staticmethod
    def minimal_coin_need(n, m, nums):

        nums += [m + 1]
        nums.sort()
        # 有 n 个可选取且无限的硬币，为了形成 1-m 所有组合需要的最少硬币个数
        if nums[0] != 1:
            return -1
        ans = sum_ = 0
        for i in range(n):
            nex = nums[i + 1] - 1
            nex = nex if nex < m else m
            x = math.ceil((nex - sum_) / nums[i])
            x = x if x >= 0 else 0
            ans += x
            sum_ += x * nums[i]
            if sum_ >= m:
                break
        return ans


class TestGeneral(unittest.TestCase):

    def test_brain_storming(self):
        bs = BrainStorming()
        n, m = 4, 20
        nums = [1, 2, 5, 10]
        assert bs.minimal_coin_need(n, m, nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
