"""

Algorithm：nim_game
Description：game_dp|winning_state|lose_state

=====================================LuoGu======================================
2197（https://www.luogu.com.cn/problem/P2197）xor_sum|classical

===================================CodeForces===================================
1396B（https://codeforces.com/problemset/problem/1396/B）greedy|game_dp


"""

import heapq


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1396b(nums):
        # 博弈greedy，大顶heapq优先选取最大数量的石头做选择
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