"""
Algorithm：lexicographical_order|lexicographical_order_rank|comb|subset|perm
Description：kth_lexicographical_order|lexicographical_order_rank|subset_lexicographical_order|kth_subset_lexicographical|comb|perm

====================================LeetCode====================================
60（https://leetcode.cn/problems/permutation-sequence/）kth_perm|lexicographical_order
440（https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/）10-tree|kth
1415（https://leetcode.cn/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）lexicographical_order|construction
1643（https://leetcode.cn/problems/kth-smallest-instructions/）lexicographical_order
1830（https://leetcode.cn/problems/minimum-number-of-operations-to-make-string-sorted/）lexicographical_order_rank|rank_order
1842（https://leetcode.cn/problems/next-palindrome-using-same-digits/）lexicographical_order|greedy
1850（https://leetcode.cn/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/）next_lexicographical_order|bubble|greedy

=====================================LuoGu======================================
P1243（https://www.luogu.com.cn/problem/P1243）kth_subset
P1338（https://www.luogu.com.cn/problem/P1338）reverse_order_pair|counter|lexicographical_order

P2524（https://www.luogu.com.cn/problem/P2524）lexicographical_order|rank_of_perm
P2525（https://www.luogu.com.cn/problem/P2525）lexicographical_order|rank_of_perm|pre_lexicographical_order

===================================CodeForces===================================
1328B（https://codeforces.com/contest/1328/problem/B）comb|lexicographical_order
1620C（https://codeforces.com/contest/1620/problem/C）reverse_thinking|lexicographical_order
1509E（https://codeforces.com/contest/1509/problem/E）lexicographical_order|kth_rank|classical

"""
from src.mathmatics.lexico_graphical_order.template import LexicoGraphicalOrder
from src.utils.fast_io import FastIO


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
