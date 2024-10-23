"""

Algorithm：nim_game|sg_theorem
Description：game_dp|winning_state|lose_state|sprague_grundy|sg_theorem

=====================================LuoGu======================================
P2197（https://www.luogu.com.cn/problem/P2197）xor_sum|classical

===================================CodeForces===================================
1396B（https://codeforces.com/contest/1396/problem/B）greed|game_dp
2004E（https://codeforces.com/problemset/problem/2004/E）sprague_grundy|sg_theorem|game


"""

from src.math.nim_game.template import Nim
from src.math.prime_factor.template import PrimeFactor
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1396b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1396/problem/B
        tag: greed|game_dp
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

    @staticmethod
    def cf_2004e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/2004/E
        tag: sprague_grundy|sg_theorem|game
        """
        ceil = 10 ** 7
        pf = PrimeFactor(ceil + 10)
        sg = [0] * (ceil + 1)
        sg[1] = 1
        tot = 1
        for i in range(3, ceil + 1):
            if pf.min_prime[i] == i:
                tot += 1
                sg[i] = tot
            else:
                sg[i] = sg[pf.min_prime[i]]

        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ans = 0
            for num in nums:
                ans ^= sg[num]
            if ans:
                ac.st("Alice")
            else:
                ac.st("Bob")
        return
