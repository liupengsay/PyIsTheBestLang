"""
Algorithm：Lyndon 分解（Duval 算法求解）、最小表示法、最大表示法
Function：用来将字符串s分解成Lyndon串s1s2s3...
Lyndon子串定义为：当且仅当s的lexicographical_order严格小于它的所有非平凡的（非平凡：非空且不同于自身）循环同构串时， s才是 Lyndon 串。

====================================LeetCode====================================
1163（https://leetcode.com/problems/last-substring-in-lexicographical-order/）brain_teaser，转化为最大表示法，利用最小表示法求最大表示法


=====================================LuoGu======================================

=====================================AcWing=====================================
158（https://www.acwing.com/problem/content/160/）字符串的最小表示法模板题


拓展：可用于求字符串s的最小表示法
"""

from src.strings.lyndon_decomposition.template import LyndonDecomposition
from src.utils.fast_io import FastIO


class Solution:

    def __init__(self):
        return

    @staticmethod
    def lc_1163(s: str) -> str:
        # 求最大表示法
        s += chr(ord("a") - 1)
        i, _ = LyndonDecomposition().max_express(s)
        return s[i:-1]

    @staticmethod
    def ac_158(ac=FastIO()):
        # 求字符串的最小表示法
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