"""
Algorithm：后缀数组
Function：生成字符串的后缀排序

====================================LeetCode====================================
1754（https://leetcode.com/problems/largest-merge-of-two-strings/）
1698（https://leetcode.com/problems/number-of-distinct-substrings-in-a-string/）经典后缀数组应用题，利用height特性

=====================================LuoGu======================================
3809（https://www.luogu.com.cn/problem/P3809）

=====================================AcWing=====================================
140（https://www.acwing.com/problem/content/142/）后缀数组模板题

Morgan and a String（https://www.hackerrank.com/challenges/morgan-and-a-string/problem?isFullScreen=true）拼接两个字符串使得字典序最小
Suffix Array（https://judge.yosupo.jp/problem/suffixarray）
1 Number of Substrings（https://judge.yosupo.jp/problem/number_of_substrings）use sa to compute number of different substring

"""
from src.strings.suffix_array.template import SuffixArray
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1754_1(word1: str, word2: str) -> str:
        # 模板：后缀数组计算后缀的字典序大小，贪心拼接两个字符串使得字典序最大
        ind = {chr(ord("a") - 1 + i): i for i in range(27)}
        word = word1 + chr(ord("a") - 1) + word2
        sa, rk, height = SuffixArray(ind).get_array(word)
        m, n = len(word1), len(word2)
        i = 0
        j = 0
        merge = ""
        while i < m and j < n:
            if rk[i] > rk[j + m + 1]:
                merge += word1[i]
                i += 1
            else:
                merge += word2[j]
                j += 1
        merge += word1[i:]
        merge += word2[j:]
        return merge

    @staticmethod
    def lc_1754_2(word1: str, word2: str) -> str:
        # 模板：贪心比较后缀的字典序大小
        merge = ""
        i = j = 0
        m, n = len(word1), len(word2)
        while i < m and j < n:
            if word1[i:] > word2[j:]:
                merge += word1[i]
                i += 1
            else:
                merge += word2[j]
                j += 1
        merge += word1[i:]
        merge += word2[j:]
        return merge

    @staticmethod
    def hr_1(ac=FastIO()):
        # 模板：拼接两个字符串使得字典序最小
        for _ in range(ac.read_int()):
            word1 = ac.read_str().lower()
            word2 = ac.read_str().lower()

            ind = {chr(ord("a") + i): i for i in range(27)}
            word = word1 + chr(ord("z") + 1) + word2 + chr(ord("z") + 1)
            sa, rk, height = SuffixArray(ind).get_array(word)
            m, n = len(word1), len(word2)
            i = 0
            j = 0
            merge = []
            while i < m and j < n:
                if rk[i] < rk[j + m + 1]:
                    merge.append(word1[i])
                    i += 1
                else:
                    merge.append(word2[j])
                    j += 1
            merge.extend(list(word1[i:]))
            merge.extend(list(word2[j:]))
            ans = "".join(merge)
            ac.st(ans.upper())

        return

    @staticmethod
    def lg_3809(ac=FastIO()):
        # 模板：计算数组的后缀排序
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
        # 模板：后缀数组模板题
        ind = {chr(ord("a") + i): i for i in range(26)}
        sa, rk, height = SuffixArray(ind).get_array(ac.read_str())
        ac.lst(sa)
        ac.lst(height)
        return

    @staticmethod
    def lc_1698(s: str) -> int:
        # 模板：经典后缀数组应用题，利用 height 特性
        ind = {chr(ord("a") + i): i for i in range(26)}
        # 高度数组的定义，所有高度之和就是相同子串的个数
        sa, rk, height = SuffixArray(ind).get_array(s)
        n = len(s)
        print(height)
        return n * (n + 1) // 2 - sum(height)