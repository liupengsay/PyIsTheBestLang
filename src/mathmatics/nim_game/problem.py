"""

算法：nim游戏也叫公平组合游戏，属于博弈论范畴
功能：用来判断游戏是否存在必胜态与必输态，博弈DP类型
题目：

=====================================LuoGu======================================
2197（https://www.luogu.com.cn/problem/P2197）有一个神奇的结论：当n堆石子的数量异或和等于0时，先手必胜，否则先手必败

===================================CodeForces===================================
1396B（https://codeforces.com/problemset/problem/1396/B）博弈贪心，使用大顶堆优先选取最大数量的石头做选择


"""

import heapq


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1396b(nums):
        # 模板：博弈贪心，使用大顶堆优先选取最大数量的石头做选择
        heapq.heapify(nums)
        order = 0
        pre = 0
        while nums:
            num = -heapq.heappop(nums)
            num -= 1
            tmp = num
            if pre:
                heapq.heappush(nums, -pre)
            pre = tmp
            order = 1 - order
        return "HL" if not order else "T"