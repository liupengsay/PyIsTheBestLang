import unittest
from collections import Counter
from itertools import permutations

from src.fast_io import FastIO
from src.mathmatics.fast_power import MatrixFastPower

"""
算法：KMP算法
功能：用来处理字符串的前缀后缀相关问题
题目：

===================================力扣===================================
214. 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
796. 旋转字符串（https://leetcode.cn/problems/rotate-string/）计算字符串是否可以旋转得到
25. 找出字符串中第一个匹配项的下标（https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/）计算子字符串第一次出现的位置
1392. 最长快乐前缀（https://leetcode.cn/problems/longest-happy-prefix/）计算最长的公共前后缀，KMP与Z函数模板题
2223. 构造字符串的总得分和（https://leetcode.cn/problems/longest-happy-prefix/）利用扩展KMP计算Z函数
6918. 包含三个字符串的最短字符串（https://leetcode.cn/problems/shortest-string-that-contains-three-strings/）kmp求字符串之间的最长公共前后缀，进行贪心拼接
2851. 字符串转换（https://leetcode.cn/problems/string-transformation/description/）使用KMP与快速幂进行转移计算，也可使用字符串哈希

===================================洛谷===================================
P3375 KMP字符串匹配（https://www.luogu.com.cn/problem/P3375）计算子字符串出现的位置，与最长公共前后缀的子字符串长度
P4391 [BOI2009]Radio Transmission 无线传输（https://www.luogu.com.cn/problem/P4391）脑经急转弯加KMP算法，最优结果为 n-pi[n-1]

================================CodeForces================================
D2. Prefix-Suffix Palindrome (Hard version)（https://codeforces.com/problemset/problem/1326/D2）利用马拉车的贪心思想贪心取前后缀，再判断剩余字符的最长前后缀回文子串
D. Prefixes and Suffixes（https://codeforces.com/contest/432/problem/D）扩展kmp与kmp结合使用计数，经典z函数与前缀函数结合应用题
E. Test（https://codeforces.com/contest/25/problem/E）kmp求字符串之间的最长公共前后缀，进行贪心拼接

================================AcWing================================

141. 周期（https://www.acwing.com/problem/content/143/）利用KMP求每个字符串前缀的最小循环节
160. 匹配统计（https://www.acwing.com/problem/content/162/）z函数模板题
3823. 寻找字符串（https://www.acwing.com/problem/content/3826/）KMP与扩展KMP即z函数应用模板题

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

    def find_longest_palindrome(self, s, pos="prefix"):
        # 计算最长前缀与最长后缀回文子串
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]


class Solution:
    def __init__(self):
        return 

    @staticmethod
    def lg_p3375(ac=FastIO()):
        # 模板：KMP字符串匹配
        s1 = ac.read_str()
        s2 = ac.read_str()
        m, n = len(s1), len(s2)
        pi = KMP().prefix_function(s2+"@"+s1)
        for i in range(n, m+n+1):
            if pi[i] == n:
                ac.st(i-n+1-n)
        ac.lst(pi[:n])
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
            a = KMP().find_longest_palindrome(s)
            s1 = mid[:a]

            a = KMP().find_longest_palindrome(s, "suffix")
            s2 = mid[-a:]
            if len(s1) > len(s2):
                ac.st(s[:i] + s1 + s[j + 1:])
            else:
                ac.st(s[:i] + s2 + s[j + 1:])
        return

    @staticmethod
    def lc_214(s: str) -> str:
        # 模板：使用 KMP 计算最长回文前缀
        k = KMP().find_longest_palindrome(s)
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

    @staticmethod
    def lc_2223(s: str) -> int:
        # 模板：z 函数计算最长公共前缀
        ans = sum(KMP().z_function(s)) + len(s)
        return ans

    @staticmethod
    def lg_p4391(ac=FastIO()):
        # 模板：计算最小的循环子串使得其不断重复包含给定字符串
        n = ac.read_int()
        s = ac.read_str()
        pi = KMP().prefix_function(s)
        ac.st(n-pi[-1])
        return

    @staticmethod
    def cf_432d(ac=FastIO()):
        # 模板：z函数与kmp算法共同使用，并使用倒序计数
        s = ac.read_str()

        n = len(s)
        z = KMP().z_function(s)
        z[0] = n
        ans = []
        for i in range(n - 1, -1, -1):
            if z[i] == n - i:
                ans.append([n - i, 0])
        z.sort()

        j = n-1
        m = len(ans)
        for i in range(m-1, -1, -1):
            x = ans[i][0]
            while j >= 0 and z[j] >= x:
                j -= 1
            ans[i][1] = n-j-1

        ac.st(m)
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def ac_141(ac=FastIO()):
        # 模板：利用KMP求每个字符串前缀的最小循环节
        ind = 0
        while True:
            n = ac.read_int()
            if not n:
                break
            s = ac.read_str()
            ind += 1
            ac.st(f"Test case #{ind}")
            pi = KMP().prefix_function(s)
            for i in range(1, n):
                if i+1 - pi[i] and (i+1) % (i+1-pi[i]) == 0 and (i+1)//(i+1-pi[i]) > 1:
                    ac.lst([i+1, (i+1)//(i+1-pi[i])])
            ac.st("")
        return

    @staticmethod
    def ac_160(ac=FastIO()):
        # 模板：z函数模板题
        n, m, q = ac.read_ints()
        s = ac.read_str()
        t = ac.read_str()
        st = t+"#"+s
        z = KMP().z_function(st)
        cnt = Counter(z[m+1:])
        for _ in range(q):
            x = ac.read_int()
            ac.st(cnt[x])
        return

    @staticmethod
    def cf_25e(ac=FastIO()):

        # 模板：kmp求字符串之间的最长公共前后缀，进行贪心拼接
        s = [ac.read_str() for _ in range(3)]

        def check(a, b):
            c = b + "#" + a
            f = KMP().prefix_function(c)
            m = len(b)
            if max(f[m:]) == m:
                return a
            x = f[-1]
            return a + b[x:]

        ind = list(range(3))
        ans = sum(len(w) for w in s)
        for item in permutations(ind, 3):
            t1, t2, t3 = [s[x] for x in item]
            cur = len(check(check(t1, t2), t3))
            if cur < ans:
                ans = cur
        ac.st(ans)
        return

    @staticmethod
    def lc_6918(aa: str, bb: str, cc: str) -> str:

        def check(a, b):
            c = b + "#" + a
            f = KMP().prefix_function(c)
            m = len(b)
            if max(f[m:]) == m:
                return a
            x = f[-1]
            return a + b[x:]

        # 模板：kmp求字符串之间的最长公共前后缀，进行贪心拼接
        s = [aa, bb, cc]
        ind = list(range(3))
        ans = "".join(s)
        for item in permutations(ind, 3):
            t1, t2, t3 = [s[x] for x in item]
            cur = check(check(t1, t2), t3)
            if len(cur) < len(ans) or (len(cur) == len(ans) and cur < ans):
                ans = cur
        return ans

    @staticmethod
    def lc_2851(s: str, t: str, k: int) -> int:
        # 模板：使用KMP与快速幂进行转移计算，也可使用字符串哈希
        n = len(s)
        mod = 10**9 + 7
        kmp = KMP()
        z = kmp.prefix_function(t + "#" + s + s)
        p = sum(z[i] == n for i in range(2 * n, 3 * n))
        q = n - p
        mat = [[p - 1, p], [q, q - 1]]
        vec = [1, 0] if z[2 * n] == n else [0, 1]
        res = MatrixFastPower().matrix_pow(mat, k, mod)
        ans = vec[0] * res[0][0] + vec[1] * res[0][1]
        return ans % mod

    @staticmethod
    def ac_3823(ac=FastIO()):
        # 模板：KMP与扩展KMP即z函数应用模板题
        kmp = KMP()
        for _ in range(ac.read_int()):
            s = ac.read_str()
            if len(s) <= 2:
                ac.st("not exist")
                continue
            pre = kmp.prefix_function(s)
            z = kmp.z_function(s)
            ans = 0
            cnt = Counter(s)
            if s[0] == s[-1] and cnt[s[0]] >= 3:
                ans = 1
            m = len(s)
            for i in range(1, m-1):
                w = pre[i]
                if z[-w] == w:
                    if w > ans:
                        ans = w
            if not ans:
                ac.st("not exist")
            else:
                ac.st(s[:ans])
        return


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
