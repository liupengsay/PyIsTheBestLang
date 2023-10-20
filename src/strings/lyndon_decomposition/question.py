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

from strings.lyndon_decomposition.template import LyndonDecomposition
from utils.fast_io import FastIO


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
