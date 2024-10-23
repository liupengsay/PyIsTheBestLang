"""
Algorithm：cantor_expands
Description：the_kth_rank_perm|the_rank_of_perm|cantor_expands

=====================================LuoGu======================================
P3014（https://www.luogu.com.cn/problem/P3014）the_kth_rank_perm|the_rank_of_perm|cantor_expands
P5367（https://www.luogu.com.cn/problem/P5367）the_kth_rank_perm|the_rank_of_perm|cantor_expands



=====================================AcWing=====================================
5052（https://www.acwing.com/problem/content/5055/）the_kth_rank_perm|the_rank_of_perm|cantor_expands|bfs


"""

import math

from src.math.cantor_expands.template import CantorExpands
from src.math.lexico_graphical_order.template import LexicoGraphicalOrder
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p5367(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5367
        tag: the_kth_rank_perm|the_rank_of_perm|cantor_expands
        """

        n = ac.read_int()  # TLE
        nums = ac.read_list_ints()
        mod = 998244353
        ce = CantorExpands(n, mod)
        ac.st(ce.array_to_rank(nums) % mod)  # array_to_rank_with_tree
        return

    @staticmethod
    def lg_p3014_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3014
        tag: the_kth_rank_perm|the_rank_of_perm|cantor_expands
        """

        n, q = ac.read_list_ints()
        ct = CantorExpands(n, mod=math.factorial(n + 2))
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(ct.rank_to_array(n, lst[0]))
            else:
                ac.st(ct.array_to_rank(lst))
        return

    @staticmethod
    def lg_p3014_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3014
        tag: the_kth_rank_perm|the_rank_of_perm|cantor_expands
        """

        n, q = ac.read_list_ints()
        og = LexicoGraphicalOrder()
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(og.get_kth_subset_perm(n, lst[0]))
            else:
                ac.st(og.get_subset_perm_kth(n, lst))
        return

    @staticmethod
    def ac_5052(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/5055/
        tag: the_kth_rank_perm|the_rank_of_perm|cantor_expands|bfs
        """

        n, k = ac.read_list_ints()
        low = max(1, n - 12)
        high = n

        if math.factorial(high - low + 1) < k:
            ac.st(-1)
            return

        lst = list(range(low, high + 1))
        m = high - low + 1
        ce = CantorExpands(m, 10 ** 20)
        ind = ce.rank_to_array(m, k)
        perm = [lst[i - 1] for i in ind]
        ans = 0
        stack = [0]
        while stack:
            x = stack.pop()
            if x > n:
                continue
            stack.append(x * 10 + 4)
            stack.append(x * 10 + 7)
            if 1 <= x <= n:
                a_x = x if x < low else perm[x - low]
                if all(w in "47" for w in str(a_x)):
                    ans += 1
        ac.st(ans)
        return
