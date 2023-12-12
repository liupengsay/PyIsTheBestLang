"""
Algorithm：manacher|palindrome_substring|plindrome_subsequence
Description：dp|center|center_expansion_method|manacher

====================================LeetCode====================================
5（https://leetcode.com/problems/longest-palindromic-substring/）longest_palindrome_substring|classical
132（https://leetcode.com/problems/palindrome-partitioning-ii/）linear_dp|manacher|longest_palindrome_substring
1960（https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/）longest_palindrome_substring|prefix_suffix|classical

=====================================LuoGu======================================
P4555（https://www.luogu.com.cn/problem/P4555）longest_palindrome_substring|prefix_suffix
P1210（https://www.luogu.com.cn/problem/P1210）longest_palindrome_substring
P4888（https://www.luogu.com.cn/problem/P4888）center_expansion_method|two_pointers
P1872（https://www.luogu.com.cn/problem/P1872）counter|palindrome_substring|manacher|classical
P6297（https://www.luogu.com.cn/problem/P6297）center_expansion_method

===================================CodeForces===================================
1682A（https://codeforces.com/contest/1682/problem/A）palindromic|center_extension
139（https://www.acwing.com/problem/content/141/）manacher|longest_palindrome_substring|binary_search|hash

===================================LibraryChecker===================================
1 Enumerate Palindromes（https://judge.yosupo.jp/problem/enumerate_palindromes）counter|palindrome_substring

"""

from src.strings.manacher_palindrome.template import ManacherPlindrome

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        s = ac.read_str()
        t = "#".join(s)
        dp = ManacherPlindrome().manacher(t)
        n = len(t)

        for i in range(n):
            x = 2 * dp[i] - 1
            if t[i + dp[i] - 1] == "#":
                dp[i] = (x - 1) // 2
            else:
                dp[i] = (x + 1) // 2
        ac.lst(dp)
        return

    @staticmethod
    def lc_1745(s: str) -> bool:
        # matrix_dp判断是否为palindrome_substring，或者manacher然后brute_force
        start, end = ManacherPlindrome().palindrome(s)
        dct = [set(ls) for ls in end]
        for i in start[0]:
            for j in end[-1]:
                if i < j and i + 1 in dct[j - 1]:
                    return True
        return False

    @staticmethod
    def lg_4555(s):
        # 长度和最大的两个palindrome_substring的长度和，转换为求字符开头以及结尾的最长palindrome_substring
        n = len(s)
        post, pre = ManacherPlindrome().palindrome_longest(s)
        ans = max(post[i + 1] + pre[i] for i in range(n - 1))
        return ans

    @staticmethod
    def lc_5(s: str) -> str:
        """
        url: https://leetcode.com/problems/longest-palindromic-substring/
        tag: longest_palindrome_substring|classical
        """
        # 字符串的最长palindrome_substring，转换为求字符开头以及结尾的最长palindrome_substring
        post, pre = ManacherPlindrome().palindrome_longest(s)
        i = post.index(max(post))
        return s[i: i + post[i]]

    @staticmethod
    def ac_139(ac=FastIO()):
        # manacher最长palindrome_substring的长度
        ind = 0
        while True:
            s = ac.read_str()
            if s == "END":
                break
            ind += 1
            ans = ManacherPlindrome().palindrome_longest_length(s)
            ac.st(f"Case {ind}: {ans}")
        return

    @staticmethod
    def lg_p1876(ac=FastIO()):
        # 回文串对数统计，利用manacher以当前字母开头与结尾的回文串数
        s = ac.read_str()
        n = len(s)
        start, end = ManacherPlindrome().palindrome(s)
        start = [len(x) for x in start]
        end = [len(x) for x in end]
        pre = ans = 0
        for i in range(n):
            ans += pre * start[i]
            pre += end[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6297(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6297
        tag: center_expansion_method
        """
        # center_expansion_method并变量维护
        n, k = ac.read_list_ints()
        mod = 10 ** 9 + 7
        nums = ac.read_list_ints()
        ans = 0
        for i in range(n):

            cur = nums[i]
            rem = k
            x, y = i - 1, i + 1
            while x >= 0 and y < n:
                if nums[x] != nums[y]:
                    if not rem:
                        break
                    rem -= 1
                cur *= nums[x] * nums[y]
                x -= 1
                y += 1
            ans = ac.max(ans, cur)

            if i + 1 < n:
                cur = 0
                rem = k
                x, y = i, i + 1
                while x >= 0 and y < n:
                    if nums[x] != nums[y]:
                        if not rem:
                            break
                        rem -= 1
                    cur = cur if cur else 1
                    cur *= nums[x] * nums[y]
                    x -= 1
                    y += 1
                ans = ac.max(ans, cur)
        ac.st(ans % mod)
        return

    @staticmethod
    def lc_1960(s: str) -> int:
        """
        url: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/
        tag: longest_palindrome_substring|prefix_suffix|classical
        """
        # 利用manacher求解每个位置前后最长palindrome_substring
        post, pre = ManacherPlindrome().palindrome_longest(s)

        n = len(s)
        if post[-1] % 2 == 0:
            post[-1] = 1
        for i in range(n - 2, -1, -1):
            if post[i] % 2 == 0:
                post[i] = 1
            post[i] = post[i] if post[i] > post[i + 1] else post[i + 1]

        ans = x = 0
        for i in range(n - 1):
            if pre[i] % 2 == 0:
                pre[i] = 1
            x = x if x > pre[i] else pre[i]
            ans = ans if ans > x * post[i + 1] else x * post[i + 1]
        return ans

    @staticmethod
    def lg_p1782(ac=FastIO()):
        # 回文串对数统计，利用manacher以当前字母开头与结尾的回文串数
        s = ac.read_str()
        n = len(s)
        start, end = ManacherPlindrome().palindrome(s)
        start = [len(x) for x in start]
        end = [len(x) for x in end]
        pre = ans = 0
        for i in range(n):
            ans += pre * start[i]
            pre += end[i]
        ac.st(ans)
        return

    @staticmethod
    def lc_2472(s: str, k: int) -> int:
        # preprocess线性palindrome_substring DP 优化外|结果linear_dp 也可以manacher回文串获取回文信息
        n = len(s)
        _, end = ManacherPlindrome().palindrome(s)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in end[i]:
                if i - j + 1 >= k and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return dp[n]