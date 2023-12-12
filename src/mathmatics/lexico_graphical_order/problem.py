"""
Algorithm：lexicographical_order|lexicographical_order_rank|comb|subset|perm
Description：kth_lexicographical_order|lexicographical_order_rank|subset_lexicographical_order|kth_subset_lexicographical|comb|perm

====================================LeetCode====================================
60（https://leetcode.com/problems/permutation-sequence/）kth_perm|lexicographical_order
440（https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/）10-tree|kth
1415（https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）lexicographical_order|construction
1643（https://leetcode.com/problems/kth-smallest-instructions/）lexicographical_order
1830（https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/）lexicographical_order_rank|rank_order
1842（https://leetcode.com/problems/next-palindrome-using-same-digits/）lexicographical_order|greedy
1850（https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/）next_lexicographical_order|bubble|greedy

=====================================LuoGu======================================
1243（https://www.luogu.com.cn/problem/P1243）kth_subset
1338（https://www.luogu.com.cn/problem/P1338）reverse_order_pair|counter|lexicographical_order

2524（https://www.luogu.com.cn/problem/P2524）lexicographical_order|rank_of_perm
2525（https://www.luogu.com.cn/problem/P2525）lexicographical_order|rank_of_perm|pre_lexicographical_order

===================================CodeForces===================================
1328B（https://codeforces.com/problemset/problem/1328/B）comb|lexicographical_order
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
        # comb选取的lexicographical_order
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            ind = LexicoGraphicalOrder().get_kth_subset_comb(n, 2, n * (n - 1) // 2 - k + 1)
            ans = ["a"] * n
            for i in ind:
                ans[i - 1] = "b"
            ac.st("".join(ans))
        return

    @staticmethod
    def lc_440(n, k):
        """
        url: https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/
        tag: 10-tree|kth
        """
        #  1 到 n lexicographical_order第 k 小的数字
        return LexicoGraphicalOrder().get_kth_num(n, k)

    @staticmethod
    def lg_p1243(ac=FastIO()):
        # 获取第 k 小的子集
        n, k = ac.read_list_ints()
        lst = LexicoGraphicalOrder().get_kth_subset(n, k)
        ac.lst(lst)
        return

    @staticmethod
    def lg_p2524(ac=FastIO()):
        #  1 到 n 的全排列中 lst 的lexicographical_order排名
        n = ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        rk = LexicoGraphicalOrder().get_subset_perm_kth(n, lst)
        ac.st(rk)
        return

    @staticmethod
    def lg_p3014(ac=FastIO()):

        # 康托展开也可以lexicographical_ordergreedy
        n, q = ac.read_list_ints()
        og = LexicoGraphicalOrder()
        # ct = CantorExpands(n, mod=math.factorial(n + 2))
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                ac.lst(og.get_kth_subset_perm(n, lst[0]))
                # ac.lst(ct.rank_to_array(n, lst[0]))
            else:
                ac.st(og.get_subset_perm_kth(n, lst))
                # ac.st(ct.array_to_rank(lst))
        return

    @staticmethod
    def lc_60(n: int, k: int) -> str:
        """
        url: https://leetcode.com/problems/permutation-sequence/
        tag: kth_perm|lexicographical_order
        """
        #  全排列的第 k 个排列
        ans = LexicoGraphicalOrder().get_kth_subset_perm(n, k)
        return "".join(str(x) for x in ans)