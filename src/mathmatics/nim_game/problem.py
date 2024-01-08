"""

Algorithm：nim_game
Description：game_dp|winning_state|lose_state

=====================================LuoGu======================================
P2197（https://www.luogu.com.cn/problem/P2197）xor_sum|classical

===================================CodeForces===================================
1396B（https://codeforces.com/contest/1396/problem/B）greedy|game_dp


"""

from src.mathmatics.nim_game.template import Nim
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1396b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1396/problem/B
        tag: greedy|game_dp
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ceil = max(nums)
            s = sum(nums)
            if ceil > s - ceil or s % 2:
                ac.st("T")
            else:
                ac.st("HL")
        return

    @staticmethod
    def lg_p2197(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2197
        tag: xor_sum|classical
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            lst = ac.read_list_ints()
            nim = Nim(lst)
            if nim.gen_result():
                ans = "Yes"
            else:
                ans = "No"
            ac.st(ans)
        return
