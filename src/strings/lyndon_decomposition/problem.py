"""
Algorithm：lyndon_composition|minimum_expression|maximum_expression
Description：rotate_string|lexicographical_order

====================================LeetCode====================================
1163（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）brain_teaser|maximum_expression|minimum_expression|maximum_expression

=====================================LuoGu======================================
P1368（https://www.luogu.com.cn/problem/P1368）lyndon_decomposition|min_express

===================================CodeForces===================================
496B（https://codeforces.com/problemset/problem/496/B）lyndon_decomposition|min_express

=====================================AcWing=====================================
158（https://www.acwing.com/problem/content/160/）minimum_expression

=====================================LibraryChecker=====================================
1（https://codeforces.com/gym/103585/problem/K） lyndon_decomposition|max_express

"""

from src.strings.lyndon_decomposition.template import LyndonDecomposition
from src.utils.fast_io import FastIO


class Solution:

    def __init__(self):
        return

    @staticmethod
    def lc_1163(s: str) -> str:
        """
        url: https://leetcode.cn/problems/last-substring-in-lexicographical-order/
        tag: brain_teaser|maximum_expression|minimum_expression|maximum_expression
        """
        ld = LyndonDecomposition()
        ans = ld.max_express(s + "0")[0]
        return s[ans:]

    @staticmethod
    def ac_158(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/160/
        tag: minimum_expression
        """
        # 求字符串的minimum_expression
        s = ac.read_str()
        t = ac.read_str()
        _, s1 = LyndonDecomposition().min_express(s)
        _, t1 = LyndonDecomposition().min_express(t)
        if s1 == t1:
            ac.yes()
            ac.st(s1)
        else:
            ac.no()
        return

    @staticmethod
    def lg_p1368(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1368
        tag: lyndon_decomposition|min_express
        """
        ac.read_int()
        lst = ac.read_list_ints()
        ans = LyndonDecomposition().min_express(lst)
        ac.lst(ans[1])
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://codeforces.com/gym/103585/problem/K
        tag: lyndon_decomposition|max_express
        """
        s = ac.read_str()
        ans = LyndonDecomposition().max_express(s[::-1])
        if ans[0]:
            ac.st(ans[1])
        else:
            w = max(s[:-1])
            i = s.index(w)
            ac.st(s[:i + 1][::-1] + s[i + 1:][::-1])
        return

    @staticmethod
    def cf_496b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/496/B
        tag: lyndon_decomposition|min_express
        """
        ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        ld = LyndonDecomposition()
        ans = ld.min_express(lst)[1]
        for _ in range(10):
            lst = [(x + 1) % 10 for x in lst]
            cur = ld.min_express(lst)[1]
            if cur < ans:
                ans = cur
        ac.st("".join(str(x) for x in ans))
        return
