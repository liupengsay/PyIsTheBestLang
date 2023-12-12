"""
Algorithm：suffix_array
Description：suffix_array

====================================LeetCode====================================
1754（https://leetcode.cn/problems/largest-range_merge_to_disjoint-of-two-strings/）largest|suffix_array
1698（https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/）suffix_array|height

=====================================LuoGu======================================
P3809（https://www.luogu.com.cn/problem/P3809）suffix_array

=====================================AcWing=====================================
140（https://www.acwing.com/problem/content/142/）suffix_array|template

Morgan and a String（https://www.hackerrank.com/challenges/morgan-and-a-string/problem?isFullScreen=true）smallest|lexicographical_order|classical
Suffix Array（https://judge.yosupo.jp/problem/suffixarray）suffix_array
1 Number of Substrings（https://judge.yosupo.jp/problem/number_of_substrings）suffix_array|sa

"""
from src.strings.suffix_array.template import SuffixArray
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1754_1(word1: str, word2: str) -> str:
        """
        url: https://leetcode.cn/problems/largest-range_merge_to_disjoint-of-two-strings/
        tag: largest|suffix_array
        """
        # 后缀数组后缀的lexicographical_order大小，greedy拼接两个字符串使得lexicographical_order最大
        ind = {chr(ord("a") - 1 + i): i for i in range(27)}
        word = word1 + chr(ord("a") - 1) + word2
        sa, rk, height = SuffixArray(ind).get_array(word)
        m, n = len(word1), len(word2)
        i = 0
        j = 0
        range_merge_to_disjoint = ""
        while i < m and j < n:
            if rk[i] > rk[j + m + 1]:
                range_merge_to_disjoint += word1[i]
                i += 1
            else:
                range_merge_to_disjoint += word2[j]
                j += 1
        range_merge_to_disjoint += word1[i:]
        range_merge_to_disjoint += word2[j:]
        return range_merge_to_disjoint

    @staticmethod
    def lc_1754_2(word1: str, word2: str) -> str:
        """
        url: https://leetcode.cn/problems/largest-range_merge_to_disjoint-of-two-strings/
        tag: largest|suffix_array
        """
        # greedy比较后缀的lexicographical_order大小
        range_merge_to_disjoint = ""
        i = j = 0
        m, n = len(word1), len(word2)
        while i < m and j < n:
            if word1[i:] > word2[j:]:
                range_merge_to_disjoint += word1[i]
                i += 1
            else:
                range_merge_to_disjoint += word2[j]
                j += 1
        range_merge_to_disjoint += word1[i:]
        range_merge_to_disjoint += word2[j:]
        return range_merge_to_disjoint

    @staticmethod
    def hr_1(ac=FastIO()):
        # 拼接两个字符串使得lexicographical_order最小
        for _ in range(ac.read_int()):
            word1 = ac.read_str().lower()
            word2 = ac.read_str().lower()

            ind = {chr(ord("a") + i): i for i in range(27)}
            word = word1 + chr(ord("z") + 1) + word2 + chr(ord("z") + 1)
            sa, rk, height = SuffixArray(ind).get_array(word)
            m, n = len(word1), len(word2)
            i = 0
            j = 0
            range_merge_to_disjoint = []
            while i < m and j < n:
                if rk[i] < rk[j + m + 1]:
                    range_merge_to_disjoint.append(word1[i])
                    i += 1
                else:
                    range_merge_to_disjoint.append(word2[j])
                    j += 1
            range_merge_to_disjoint.extend(list(word1[i:]))
            range_merge_to_disjoint.extend(list(word2[j:]))
            ans = "".join(range_merge_to_disjoint)
            ac.st(ans.upper())

        return

    @staticmethod
    def lg_3809(ac=FastIO()):
        # 数组的后缀sorting
        words = [str(x) for x in range(10)] + [chr(i + ord("A")) for i in range(26)] + [chr(i + ord("a")) for i in
                                                                                        range(26)]
        ind = {st: i for i, st in enumerate(words)}
        s = ac.read_str()
        sa = SuffixArray(ind)
        ans, _, _ = sa.get_array(s)
        ac.lst([x + 1 for x in ans])
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        s = ac.read_str()
        ind = {chr(ord("a") + i): i for i in range(26)}
        sa, rk, height = SuffixArray(ind).get_array(s)
        n = len(s)
        ans = sum(height)
        ac.st(n * (n + 1) // 2 - ans)
        return

    @staticmethod
    def ac_140(ac=FastIO()):
        # 后缀数组模板题
        ind = {chr(ord("a") + i): i for i in range(26)}
        sa, rk, height = SuffixArray(ind).get_array(ac.read_str())
        ac.lst(sa)
        ac.lst(height)
        return

    @staticmethod
    def lc_1698(s: str) -> int:
        """
        url: https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/
        tag: suffix_array|height
        """
        # 后缀数组应用题，利用 height 特性
        ind = {chr(ord("a") + i): i for i in range(26)}
        # 高度数组的定义，所有高度之和就是相同子串的个数
        sa, rk, height = SuffixArray(ind).get_array(s)
        n = len(s)
        print(height)
        return n * (n + 1) // 2 - sum(height)