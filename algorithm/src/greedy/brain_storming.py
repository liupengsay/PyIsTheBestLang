"""

"""
"""
算法：贪心、逆向思维、抽屉原理、鸽巢原理、容斥原理
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
P2773 漂亮字串（https://www.luogu.com.cn/problem/P2773）分类讨论进行贪心
P2255 [USACO14JAN]Recording the Moolympics S（https://www.luogu.com.cn/problem/P2255）两个指针进行贪心
P2327 [SCOI2005]扫雷（https://www.luogu.com.cn/problem/P2327）脑筋急转弯进行枚举
P2777 [AHOI2016初中组]自行车比赛（https://www.luogu.com.cn/problem/P2777）贪心枚举最佳得分组合，加前后缀记录最大值

P2649 游戏预言（https://www.luogu.com.cn/problem/P2649）贪心，输的时候输最惨，赢的时候微弱优势

P1367 蚂蚁（https://www.luogu.com.cn/problem/P1367）脑筋急转弯，蚂蚁的相对移动位置排序还是不变
P1362 兔子数（https://www.luogu.com.cn/problem/P1362）找规律之后，进行广度优先搜索枚举
P1090 [NOIP2004 提高组] 合并果子 / [USACO06NOV] Fence Repair G（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）从小到大贪心合并
P1334 瑞瑞的木板（https://www.luogu.com.cn/problem/P1334）逆向思维的合并果子，从小到大合并
P1325 雷达安装（https://www.luogu.com.cn/problem/P1325）排序后进行贪心修建更新

P1250 种树（https://www.luogu.com.cn/problem/P1250）区间的贪心题，使用线段树修改区间与查询和，以及二分进行计算
P1230 智力大冲浪（https://www.luogu.com.cn/problem/P1230）排序后进行选取贪心
P1159 排行榜（https://www.luogu.com.cn/problem/P1159）使用队列贪心进行模拟
P1095 [NOIP2007 普及组] 守望者的逃离（https://www.luogu.com.cn/problem/P1095）贪心模拟也可以理解为动态规划转移

P1056 [NOIP2008 普及组] 排座椅（https://www.luogu.com.cn/record/list?user=739032&status=12&page=14）根据题意进行计数排序贪心选择
625. 最小因式分解（https://leetcode.cn/problems/minimum-factorization/）贪心进行因式分解，类似质因数分解
P8847 [JRKSJ R5] 1-1 A（https://www.luogu.com.cn/problem/P8847）分类讨论和贪心进行
P8845 [传智杯 #4 初赛] 小卡和质数（https://www.luogu.com.cn/problem/solution/P8845）脑筋急转弯，只有2是偶数质数
P2772 寻找平面上的极大点（https://www.luogu.com.cn/problem/P2772）按照两个维度排序，再按照其中一个维度顺序比较最大值

P2878 [USACO07JAN] Protecting the Flowers S（https://www.luogu.com.cn/problem/P2878）经典贪心题目，可使用举例两个计算、再进行归纳确定排序规则
P2920 [USACO08NOV]Time Management S（https://www.luogu.com.cn/problem/P2920）排序后进行贪心计算
P2983 [USACO10FEB]Chocolate Buying S（https://www.luogu.com.cn/problem/P2983）看起来是背包其实是贪心优先选择最便宜的奶牛满足
P3173 [HAOI2009]巧克力（https://www.luogu.com.cn/problem/P3173）从大到小排序进行贪心计算
P5098 [USACO04OPEN]Cave Cows 3（https://www.luogu.com.cn/problem/P5098）贪心按照一个维度排序后再按照另一个维度分类讨论，记录前缀最小值
P5159 WD与矩阵（https://www.luogu.com.cn/problem/P5159）利用异或的特点枚举计数并进行快速幂计算


P5497 [LnOI2019SP]龟速单项式变换(SMT)（https://www.luogu.com.cn/problem/P5497）抽屉原理进行分类讨论
P5682 [CSP-J2019 江西] 次大值（https://www.luogu.com.cn/problem/P5682）脑筋急转弯进行排序后贪心枚举确定
P5804 [SEERC2019]Absolute Game（https://www.luogu.com.cn/problem/P5804）排序贪心枚举和二分查找优化
P5963 [BalticOI ?] Card 卡牌游戏【来源请求】（https://www.luogu.com.cn/problem/P5963）经典贪心题目，可使用举例两个计算、再进行归纳确定排序规则
P6023 走路（https://www.luogu.com.cn/problem/P6023）可证明集中在某天是最佳结果，然后使用指针进行模拟计算
P6243 [USACO06OPEN]The Milk Queue G（https://www.luogu.com.cn/problem/P6243）经典贪心举例之后进行自定义排序


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
