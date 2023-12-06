"""
算法：马拉车算法、回文连续子串、回文不连续子串
功能：用来处理字符串的回文相关问题，可以有暴力、DP、中心扩展法、马拉车
题目：

====================================LeetCode====================================
5（https://leetcode.com/problems/longest-palindromic-substring/）计算字符串的最长回文连续子串
132（https://leetcode.com/problems/palindrome-partitioning-ii/）经典线性 DP 与马拉车判断以每个位置为结尾的回文串
1960（https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/）利用马拉车求解每个位置前后最长回文子串

=====================================LuoGu======================================
4555（https://www.luogu.com.cn/problem/P4555）计算以当前索引为开头以及结尾的最长回文子串
1210（https://www.luogu.com.cn/problem/P1210）寻找最长的连续回文子串
4888（https://www.luogu.com.cn/problem/P4888）中心扩展法双指针
1872（https://www.luogu.com.cn/problem/P1872）回文串对数统计，利用马拉车计算以当前字母开头与结尾的回文串数
6297（https://www.luogu.com.cn/problem/P6297）中心扩展法并使用变量维护

===================================CodeForces===================================
1682A（https://codeforces.com/contest/1682/problem/A）palindromic|center_extension
139（https://www.acwing.com/problem/content/141/）马拉车计算最长回文子串长度，也可使用二分查找加哈希

===================================LibraryChecker===================================
1 Enumerate Palindromes（https://judge.yosupo.jp/problem/enumerate_palindromes）

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
        # 模板：经典矩阵DP判断是否为回文子串，或者使用马拉车然后枚举
        start, end = ManacherPlindrome().palindrome(s)
        dct = [set(ls) for ls in end]
        for i in start[0]:
            for j in end[-1]:
                if i < j and i + 1 in dct[j - 1]:
                    return True
        return False

    @staticmethod
    def lg_4555(s):
        # 模板：计算长度和最大的两个回文子串的长度和，转换为求字符开头以及结尾的最长回文子串
        n = len(s)
        post, pre = ManacherPlindrome().palindrome_longest(s)
        ans = max(post[i + 1] + pre[i] for i in range(n - 1))
        return ans

    @staticmethod
    def lc_5(s: str) -> str:
        # 模板：计算字符串的最长回文子串，转换为求字符开头以及结尾的最长回文子串
        post, pre = ManacherPlindrome().palindrome_longest(s)
        i = post.index(max(post))
        return s[i: i + post[i]]

    @staticmethod
    def ac_139(ac=FastIO()):
        # 模板：马拉车计算最长回文子串的长度
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
        # 模板：回文串对数统计，利用马拉车计算以当前字母开头与结尾的回文串数
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
        # 模板：中心扩展法并使用变量维护
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
        # 模板：利用马拉车求解每个位置前后最长回文子串
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
        # 模板：回文串对数统计，利用马拉车计算以当前字母开头与结尾的回文串数
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
        # 模板：预处理线性回文子串 DP 优化外加结果计算线性 DP 也可以使用马拉车回文串获取回文信息
        n = len(s)
        _, end = ManacherPlindrome().palindrome(s)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in end[i]:
                if i - j + 1 >= k and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return dp[n]