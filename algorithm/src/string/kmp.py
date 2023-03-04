import unittest
from algorithm.src.fast_io import FastIO

"""
算法：KMP算法
功能：用来处理字符串的前缀后缀相关问题
题目：

===================================力扣===================================
214. 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
796. 旋转字符串（https://leetcode.cn/problems/rotate-string/）计算字符串是否可以旋转得到
25. 找出字符串中第一个匹配项的下标（https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/）计算子字符串第一次出现的位置
1392. 最长快乐前缀（https://leetcode.cn/problems/longest-happy-prefix/）计算最长的公共前后缀
2223. 构造字符串的总得分和（https://leetcode.cn/problems/longest-happy-prefix/）利用扩展KMP计算Z函数

===================================洛谷===================================
P3375 KMP字符串匹配（https://www.luogu.com.cn/problem/P3375）计算子字符串出现的位置，与最长公共前后缀的子字符串长度
P4391 无线传输（https://www.luogu.com.cn/problem/P4391）脑经急转弯加KMP算法，最优结果为 n-pi[n-1]


================================CodeForces================================
D2. Prefix-Suffix Palindrome (Hard version)（https://codeforces.com/problemset/problem/1326/D2）利用马拉车的贪心思想贪心取前后缀，再判断剩余字符的最长前后缀回文子串


参考：OI WiKi（https://oi-wiki.org/string/kmp/）

"""


class KMP:
    def __init__(self):
        return

    @staticmethod
    def prefix_function(s):
        # 计算s[:i]与s[:i]的最长公共真前缀与真后缀
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        # pi[0] = 0
        return pi

    @staticmethod
    def z_function(s):
        # 计算 s[i:] 与 s 的最长公共前缀
        n = len(s)
        z = [0] * n
        left, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - left] < r - i + 1:
                z[i] = z[i - left]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                left = i
                r = i + z[i] - 1
        # z[0] = 0
        return z

    def find(self, s1, s2):
        # 查找 s2 在 s1 中的索引位置
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_longest_palidrome(self, s, pos="prefix"):
        # 计算最长前缀与最长后缀回文子串
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]


class Solution:
    def __init__(self):
        return 
    
    @staticmethod
    def cf_1326d2(ac=FastIO()):
        # 模板：使用 KMP 计算最长回文前缀与后缀
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            i, j = 0, n - 1
            while i < j:
                if s[i] == s[j]:
                    i += 1
                    j -= 1
                else:
                    break
            if i >= j:
                ac.st(s)
                continue

            mid = s[i:j + 1]
            a = KMP().find_longest_palidrome(s)
            s1 = mid[:a]

            a = KMP().find_longest_palidrome(s, "suffix")
            s2 = mid[-a:]
            if len(s1) > len(s2):
                ac.st(s[:i] + s1 + s[j + 1:])
            else:
                ac.st(s[:i] + s2 + s[j + 1:])
        return

    @staticmethod
    def lc_214(s: str) -> str:
        # 模板：使用 KMP 计算最长回文前缀
        k = KMP().find_longest_palidrome(s)
        return s[k:][::-1] + s

    @staticmethod
    def lc_796(s: str, goal: str) -> bool:
        ans = KMP().find(s+s, goal)
        return len(ans) > 0 and len(s) == len(goal)

    @staticmethod
    def lc_28(haystack: str, needle: str) -> int:
        ans = KMP().find(haystack, needle)
        return ans[0] if ans else -1

    @staticmethod
    def lc_1392(s: str) -> str:
        # 模板：字符串的最长非空真前缀（同时也是非空真后缀）
        lst = KMP().prefix_function(s)
        return s[:lst[-1]]


class TestGeneral(unittest.TestCase):

    def test_prefix_function(self):
        kmp = KMP()
        assert kmp.prefix_function("abcabcd") == [0, 0, 0, 1, 2, 3, 0]
        assert kmp.prefix_function("aabaaab") == [0, 1, 0, 1, 2, 2, 3]
        return

    def test_z_function(self):
        kmp = KMP()
        assert kmp.z_function("abacaba") == [0, 0, 1, 0, 3, 0, 1]
        assert kmp.z_function("aaabaab") == [0, 2, 1, 0, 2, 1, 0]
        assert kmp.z_function("aaaaa") == [0, 4, 3, 2, 1]
        return

    def test_find(self):
        kmp = KMP()
        assert kmp.find("abababc", "aba") == [0, 2]
        assert kmp.find("aaaa", "a") == [0, 1, 2, 3]
        assert kmp.find("aaaa", "aaaa") == [0]
        return


if __name__ == '__main__':
    unittest.main()
