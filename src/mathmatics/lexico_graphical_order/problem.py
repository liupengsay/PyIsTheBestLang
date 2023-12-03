"""
算法：字典序与字典序排名解析
功能：计算字典序第K小和某个对象的字典序rank、计算subset的字典序与解析、计算comb的字典序与解析、计算perm的字典序与解析
题目：

===================================力扣===================================
60. 排列序列（https://leetcode.com/problems/permutation-sequence/）全排列的第 k 个排列
440. 字典序的第K小数字（https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/）经典面试题使用十叉树求解
1415. 长度为 n 的开心字符串中字典序第 k 小的字符串（https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/）类似思路经典字典序构造
1643. 第 K 条最小指令（https://leetcode.com/problems/kth-smallest-instructions/）根据字典序进行处理
1830. 使字符串有序的最少操作次数（https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/）经典计算字符串的字典序 rank，也可以反向计算rank的字典序
1842. 下个由相同数字构成的回文串（https://leetcode.com/problems/next-palindrome-using-same-digits/）经典字典序贪心
1850. 邻位交换的最小次数（https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/）经典下一个字典序与冒泡贪心交换次数

===================================洛谷===================================
P1243 排序集合（https://www.luogu.com.cn/problem/P1243）求出第K小的子集
P1338 末日的传说（https://www.luogu.com.cn/problem/P1338）结合逆序对计数的字典序

P2524 Uim的情人节礼物·其之弐（https://www.luogu.com.cn/problem/P2524）计算全排列的字典序排名
P2525 Uim的情人节礼物·其之壱（https://www.luogu.com.cn/problem/P2525）计算全排列的上一个排列
P3014 [USACO11FEB]Cow Line S（https://www.luogu.com.cn/problem/P3014）计算全排列的排名与排名对应的全排列

================================CodeForces================================
B. K-th Beautiful String（https://codeforces.com/problemset/problem/1328/B）计算comb的字典序
C. BA-String（https://codeforces.com/contest/1620/problem/C）经典逆向思维字典序
E. Almost Sorted（https://codeforces.com/contest/1509/problem/E）字典序典题，计算rank k的数组

参考：OI WiKi（xx）
"""
from src.mathmatics.lexico_graphical_order.template import LexicoGraphicalOrder
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1328b(ac=FastIO()):
        # 模板：计算comb选取的字典序
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
        # 模板：计算 1 到 n 字典序第 k 小的数字
        return LexicoGraphicalOrder().get_kth_num(n, k)

    @staticmethod
    def lg_p1243(ac=FastIO()):
        # 模板：获取第 k 小的子集
        n, k = ac.read_list_ints()
        lst = LexicoGraphicalOrder().get_kth_subset(n, k)
        ac.lst(lst)
        return

    @staticmethod
    def lg_p2524(ac=FastIO()):
        # 模板：计算 1 到 n 的全排列中 lst 的字典序排名
        n = ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        rk = LexicoGraphicalOrder().get_subset_perm_kth(n, lst)
        ac.st(rk)
        return

    @staticmethod
    def lg_p3014(ac=FastIO()):

        # 模板：康托展开也可以使用字典序贪心计算
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
        #  模板：全排列的第 k 个排列
        ans = LexicoGraphicalOrder().get_kth_subset_perm(n, k)
        return "".join(str(x) for x in ans)
