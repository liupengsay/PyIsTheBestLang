"""
Algorithm：lexicographical_order|lexicographical_order_rank|comb|subset|perm
Description：kth_lexicographical_order|lexicographical_order_rank|subset_lexicographical_order|kth_subset_lexicographical|comb|perm

====================================LeetCode====================================
31（https://leetcode.cn/problems/next-permutation/）next_permutation|classical
60（https://leetcode.cn/problems/permutation-sequence/）kth_perm|lexicographical_order
440（https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/）10-tree|kth
1415（https://leetcode.cn/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）lexicographical_order|construction
1643（https://leetcode.cn/problems/kth-smallest-instructions/）lexicographical_order
1830（https://leetcode.cn/problems/minimum-number-of-operations-to-make-string-sorted/）lexicographical_order_rank|rank_order
1842（https://leetcode.cn/problems/next-palindrome-using-same-digits/）lexicographical_order|greed
1850（https://leetcode.cn/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/）next_lexicographical_order|bubble|greed

=====================================LuoGu======================================
P1243（https://www.luogu.com.cn/problem/P1243）kth_subset
P1338（https://www.luogu.com.cn/problem/P1338）reverse_order_pair|counter|lexicographical_order
P2524（https://www.luogu.com.cn/problem/P2524）lexicographical_order|rank_of_perm
P2525（https://www.luogu.com.cn/problem/P2525）lexicographical_order|rank_of_perm|pre_lexicographical_order

=====================================AtCoder======================================
ABC276C（https://atcoder.jp/contests/abc276/tasks/abc276_c）prev_permutation|classical
ABC202D（https://atcoder.jp/contests/abc202/tasks/abc202_d）lexicographical_order|rank|construction

===================================CodeForces===================================
1328B（https://codeforces.com/contest/1328/problem/B）comb|lexicographical_order
1620C（https://codeforces.com/contest/1620/problem/C）reverse_thinking|lexicographical_order
1509E（https://codeforces.com/contest/1509/problem/E）lexicographical_order|kth_rank|classical

"""
from typing import List

from src.math.lexico_graphical_order.template import LexicoGraphicalOrder, Permutation
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1328b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1328/problem/B
        tag: comb|lexicographical_order
        """
        lgo = LexicoGraphicalOrder()
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            ind = lgo.get_kth_subset_comb(n, 2, n * (n - 1) // 2 - k + 1)
            ans = ["a"] * n
            for i in ind:
                ans[i - 1] = "b"
            ac.st("".join(ans))
        return

    @staticmethod
    def lc_440(n, k):
        """
        url: https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/
        tag: 10-tree|kth
        """
        return LexicoGraphicalOrder().get_kth_num(n, k)

    @staticmethod
    def lg_p1243(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1243
        tag: kth_subset
        """
        n, k = ac.read_list_ints()
        lst = LexicoGraphicalOrder().get_kth_subset(n, k)
        ac.lst(lst)
        return

    @staticmethod
    def lg_p2524(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2524
        tag: lexicographical_order|rank_of_perm
        """
        n = ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        rk = LexicoGraphicalOrder().get_subset_perm_kth(n, lst)
        ac.st(rk)
        return

    @staticmethod
    def lc_60(n: int, k: int) -> str:
        """
        url: https://leetcode.cn/problems/permutation-sequence/
        tag: kth_perm|lexicographical_order
        """
        ans = LexicoGraphicalOrder().get_kth_subset_perm(n, k)
        return "".join(str(x) for x in ans)

    @staticmethod
    def abc_276c_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc276/tasks/abc276_c
        tag: prev_permutation|classical
        """
        ac.read_int()
        lst = ac.read_list_ints()
        ac.lst(Permutation().prev_permutation(lst))
        return

    @staticmethod
    def abc_276c_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc276/tasks/abc276_c
        tag: prev_permutation|classical
        """
        n = ac.read_int()
        lst = ac.read_list_ints()
        lgo = LexicoGraphicalOrder()
        k = lgo.get_subset_perm_kth(n, lst)
        ac.lst(lgo.get_kth_subset_perm(n, k - 1))
        return

    @staticmethod
    def lc_31(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/next-permutation/
        tag: next_permutation|classical
        """
        nums = Permutation().next_permutation(nums)
        return nums
