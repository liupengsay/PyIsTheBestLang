"""
Algorithm：lyndon_composition|minimum_expression|maximum_expression
Description：rotate_string|lexicographical_order

====================================LeetCode====================================
1163（https://leetcode.com/problems/last-substring-in-lexicographical-order/）brain_teaser|maximum_expression|minimum_expression|maximum_expression

=====================================LuoGu======================================

=====================================AcWing=====================================
158（https://www.acwing.com/problem/content/160/）minimum_expression

"""

from src.strings.lyndon_decomposition.template import LyndonDecomposition
from src.utils.fast_io import FastIO


class Solution:

    def __init__(self):
        return

    @staticmethod
    def lc_1163(s: str) -> str:
        """
        url: https://leetcode.com/problems/last-substring-in-lexicographical-order/
        tag: brain_teaser|maximum_expression|minimum_expression|maximum_expression
        """
        # 求maximum_expression
        s += chr(ord("a") - 1)
        i, _ = LyndonDecomposition().max_express(s)
        return s[i:-1]

    @staticmethod
    def ac_158(ac=FastIO()):
        # 求字符串的minimum_expression
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