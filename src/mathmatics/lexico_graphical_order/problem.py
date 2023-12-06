"""
Algorithm：lexicographical_order与lexicographical_order排名解析
Function：lexicographical_order第K小和某个对象的lexicographical_orderrank、subset的lexicographical_order与解析、comb的lexicographical_order与解析、perm的lexicographical_order与解析

====================================LeetCode====================================
60（https://leetcode.com/problems/permutation-sequence/）全排列的第 k 个排列
440（https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/）面试题十叉树求解
1415（https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）类似思路lexicographical_order构造
1643（https://leetcode.com/problems/kth-smallest-instructions/）根据lexicographical_order处理
1830（https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/）字符串的lexicographical_order rank，也可以反向rank的lexicographical_order
1842（https://leetcode.com/problems/next-palindrome-using-same-digits/）lexicographical_ordergreedy
1850（https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/）下一个lexicographical_order与冒泡greedy交换次数

=====================================LuoGu======================================
1243（https://www.luogu.com.cn/problem/P1243）求出第K小的子集
1338（https://www.luogu.com.cn/problem/P1338）结合逆序对counter的lexicographical_order

2524（https://www.luogu.com.cn/problem/P2524）全排列的lexicographical_order排名
2525（https://www.luogu.com.cn/problem/P2525）全排列的上一个排列

===================================CodeForces===================================
1328B（https://codeforces.com/problemset/problem/1328/B）comb的lexicographical_order
1620C（https://codeforces.com/contest/1620/problem/C）reverse_thinkinglexicographical_order
1509E（https://codeforces.com/contest/1509/problem/E）lexicographical_order典题，rank k的数组

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
        #  全排列的第 k 个排列
        ans = LexicoGraphicalOrder().get_kth_subset_perm(n, k)
        return "".join(str(x) for x in ans)