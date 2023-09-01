import unittest

from src.fast_io import FastIO

"""
算法：Lyndon 分解（使用Duval 算法求解）、最小表示法、最大表示法
功能：用来将字符串s分解成Lyndon串s1s2s3...
Lyndon子串定义为：当且仅当s的字典序严格小于它的所有非平凡的（非平凡：非空且不同于自身）循环同构串时， s才是 Lyndon 串。
题目：

===================================力扣===================================
1163. 按字典序排在最后的子串（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）脑筋急转弯，转化为最大表示法，利用最小表示法求最大表示法


===================================洛谷===================================
P6657 【模板】LGV 引理（https://www.luogu.com.cn/problem/P6657）
参考：OI WiKi（https://oi-wiki.org/string/lyndon/）Duval 可以在 O(n)的时间内求出一个串的 Lyndon 分解

===================================AcWing=====================================
158. 项链（https://www.acwing.com/problem/content/160/）字符串的最小表示法模板题


拓展：可用于求字符串s的最小表示法
"""


class LyndonDecomposition:
    def __init__(self):
        return

    # duval_algorithm
    @staticmethod
    def solve_by_duval(s):
        n, i = len(s), 0
        factorization = []
        while i < n:
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                factorization.append(s[i: i + j - k])
                i += j - k
        return factorization

    # smallest_cyclic_string
    @staticmethod
    def min_cyclic_string(s):
        s += s
        n = len(s)
        i, ans = 0, 0
        while i < n // 2:
            ans = i
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                i += j - k
        return s[ans: ans + n // 2]

    @staticmethod
    def min_express(sec):
        # 最小表示法
        n = len(sec)
        k, i, j = 0, 0, 1
        while k < n and i < n and j < n:
            if sec[(i + k) % n] == sec[(j + k) % n]:
                k += 1
            else:
                if sec[(i + k) % n] > sec[(j + k) % n]:  # 最小表示法
                    i = i + k + 1
                else:
                    j = j + k + 1
                if i == j:
                    i += 1
                k = 0
        i = i if i < j else j
        return i, sec[i:] + sec[:i]

    @staticmethod
    def max_express(sec):
        # 最大表示法
        n = len(sec)
        k, i, j = 0, 0, 1
        while k < n and i < n and j < n:
            if sec[(i + k) % n] == sec[(j + k) % n]:
                k += 1
            else:
                if sec[(i + k) % n] < sec[(j + k) % n]:  # 最大表示法
                    i = i + k + 1
                else:
                    j = j + k + 1
                if i == j:
                    i += 1
                k = 0
        i = i if i < j else j
        return i, sec[i:] + sec[:i]


class Solution:

    def __init__(self):
        return

    @staticmethod
    def lc_1163(s: str) -> str:
        # 模板：求最大表示法
        s += chr(ord("a") - 1)
        i, _ = LyndonDecomposition().max_express(s)
        return s[i:-1]

    @staticmethod
    def ac_158(ac=FastIO()):
        # 模板：求字符串的最小表示法
        s = ac.read_str()
        t = ac.read_str()
        _, s1 = LyndonDecomposition().min_express(s)
        _, t1 = LyndonDecomposition().min_express(t)
        if s1 == t1:
            ac.st("Yes")
            ac.st(s1)
        else:
            ac.st("No")
        return


class TestGeneral(unittest.TestCase):
    def test_solve_by_duval(self):
        ld = LyndonDecomposition()
        assert ld.solve_by_duval("ababa") == ["ab", "ab", "a"]
        return

    def test_min_cyclic_string(self):
        ld = LyndonDecomposition()
        assert ld.min_cyclic_string("ababa") == "aabab"
        assert ld.min_express("ababa") == "aabab"
        return


if __name__ == '__main__':
    unittest.main()
