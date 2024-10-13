"""
Algorithm：manacher|palindrome_substring|plindrome_subsequence
Description：dp|center|center_expansion_method|manacher

====================================LeetCode====================================
5（https://leetcode.cn/problems/longest-palindromic-substring/）longest_palindrome_substring|classical
132（https://leetcode.cn/problems/palindrome-partitioning-ii/）linear_dp|manacher|longest_palindrome_substring
1960（https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-substrings/）longest_palindrome_substring|prefix_suffix|classical
1745（https://leetcode-cn.com/problems/palindrome-partitioning-iv/）manacher|palindrome_start_end
2472（https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/）
214（https://leetcode-cn.com/problems/shortest-palindrome/）manacher|linear_dp|palindrome_start_end
647（https://leetcode-cn.com/problems/palindromic-substrings/）manacher|palindrome_count

=====================================AcWing======================================
141（https://www.acwing.com/problem/content/141/）manacher|longest_palindrome_substring|binary_search|hash

=====================================LuoGu======================================
P4555（https://www.luogu.com.cn/problem/P4555）longest_palindrome_substring|prefix_suffix
P1210（https://www.luogu.com.cn/problem/P1210）longest_palindrome_substring
P4888（https://www.luogu.com.cn/problem/P4888）center_expansion_method|two_pointers
P1872（https://www.luogu.com.cn/problem/P1872）counter|palindrome_substring|manacher|classical
P6297（https://www.luogu.com.cn/problem/P6297）center_expansion_method|plindrome
P3805（https://www.luogu.com.cn/problem/P3805）palindrome_longest_length|manacher
P1659（https://www.luogu.com.cn/problem/P1659）manacher|palindrome_length_count
P3501（https://www.luogu.com.cn/problem/P3501）manacher|palindrome_length_count|classical|change_manacher
P6216（https://www.luogu.com.cn/problem/P6216）
P5446（https://www.luogu.com.cn/problem/P5446）

===================================CodeForces===================================
1682A（https://codeforces.com/contest/1682/problem/A）palindromic|center_extension
1326D2（https://codeforces.com/problemset/problem/1326/D2）palindrome_post_pre|manacher
7D（https://codeforces.com/problemset/problem/7/D）palindrome_just_start|manacher
835D（https://codeforces.com/problemset/problem/835/D）
17E（https://codeforces.com/contest/17/problem/E）palindrome_count_start_end|manacher
1081H（https://codeforces.com/problemset/problem/1081/H）
1827C（https://codeforces.com/contest/1827/problem/C）


===================================LibraryChecker===================================
1（https://judge.yosupo.jp/problem/enumerate_palindromes）counter|palindrome_substring
2（https://www.luogu.com.cn/problem/UVA11475）palindrome_just_end|manacher


"""

from src.strings.manacher_palindrome.template import ManacherPlindrome

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1745(s: str) -> bool:
        """
        url: https://leetcode-cn.com/problems/palindrome-partitioning-iv/
        tag: manacher|palindrome_start_end
        """
        start, end = ManacherPlindrome().palindrome_start_end(s)
        dct = [set(ls) for ls in end]
        for i in start[0]:
            for j in end[-1]:
                if i < j and i + 1 in dct[j - 1]:
                    return True
        return False

    @staticmethod
    def lg_4555(s):
        """
        url: https://www.luogu.com.cn/problem/P4555
        tag: longest_palindrome_substring|prefix_suffix
        """
        n = len(s)
        post, pre = ManacherPlindrome().palindrome_post_pre(s)
        ans = max(post[i + 1] + pre[i] for i in range(n - 1))
        return ans

    @staticmethod
    def lc_5(s: str) -> str:
        """
        url: https://leetcode.cn/problems/longest-palindromic-substring/
        tag: longest_palindrome_substring|classical
        """
        post, pre = ManacherPlindrome().palindrome_post_pre(s)
        i = post.index(max(post))
        return s[i: i + post[i]]

    @staticmethod
    def ac_141(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/141/
        tag: manacher|longest_palindrome_substring|binary_search|hash
        """
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
    def lg_p1872(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1872
        tag: counter|palindrome_substring|manacher|classical
        """

        s = ac.read_str()
        n = len(s)
        start, end = ManacherPlindrome().palindrome_start_end(s)
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
        tag: center_expansion_method|plindrome
        """

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
            ans = max(ans, cur)

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
                ans = max(ans, cur)
        ac.st(ans % mod)
        return

    @staticmethod
    def lc_1960(s: str) -> int:
        """
        url: https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-substrings/
        tag: longest_palindrome_substring|prefix_suffix|classical
        """
        post, pre = ManacherPlindrome().palindrome_post_pre(s)

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
    def lc_2472(s: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/
        tag: manacher|linear_dp|palindrome_start_end
        """
        n = len(s)
        _, end = ManacherPlindrome().palindrome_start_end(s)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in end[i]:
                if i - j + 1 >= k and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return dp[n]

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/enumerate_palindromes
        tag: manacher
        """
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
    def lg_p3805(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3805
        tag: palindrome_longest_length|manacher
        """
        s = ac.read_str()
        ans = ManacherPlindrome().palindrome_longest_length(s)
        ac.st(ans)
        return

    @staticmethod
    def cf_1362d2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1326/D2
        tag: palindrome_post_pre|manacher
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)

            i, j = 0, n - 1
            while i < j and s[i] == s[j]:
                i += 1
                j -= 1
            ans = s[:i]
            s = s[i:j + 1]
            post, pre = ManacherPlindrome().palindrome_post_pre(s)
            n = len(s)
            mid = ""
            for i in range(n - 1, -1, -1):
                if pre[i] == i + 1:
                    mid = s[:i + 1]
                    break
            for i in range(n):
                if post[i] == n - i:
                    if n - i > len(mid):
                        mid = s[-post[i]:]
                    break
            ac.st(ans + mid + ans[::-1])
        return

    @staticmethod
    def cf_7d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/7/D
        tag: palindrome_just_start|manacher
        """
        s = ac.read_str()
        dct = set(ManacherPlindrome().palindrome_just_start(s))
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(n):
            if i not in dct:
                continue
            dp[i + 1] = dp[(i + 1) // 2] + 1
        ac.st(sum(dp))
        return

    @staticmethod
    def cf_17e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/17/problem/E
        tag: palindrome_count_start_end|manacher|inclusion_exclusion|reverse_thinking
        """
        mod = 51123987
        n = ac.read_int()
        s = ac.read_str()
        start, end = ManacherPlindrome().palindrome_count_start_end(s)
        tot = sum(start)
        ans = tot * (tot - 1) // 2
        pre = 0
        for i in range(n):
            ans -= pre * start[i]
            pre += end[i]
            pre %= mod
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p1659(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1659
        tag: manacher|palindrome_length_count
        """
        n, k = ac.read_list_ints()
        s = ac.read_str()
        cnt = ManacherPlindrome().palindrome_length_count(s)
        ans = 1
        mod = 19930726
        for i in range(n, 0, -1):
            if i % 2:
                x = min(cnt[i], k)
                ans *= pow(i, x, mod)
                ans %= mod
                k -= x
                if not k:
                    ac.st(ans)
                    return
        ac.st(-1)
        return

    @staticmethod
    def lg_p3501(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3501
        tag: manacher|palindrome_length_count|classical|change_manacher
        """

        def manacher(s):
            """template of get the palindrome radius for every i-th character as center"""
            n = len(s)
            arm = [0] * n
            left, right = 0, -1
            for i in range(0, n):
                a, b = arm[left + right - i], right - i + 1
                a = a if a < b else b
                k = 0 if i > right else a
                while 0 <= i - k and i + k < n and (
                        (s[i - k] != s[i + k] and s[i - k] != "#" and s[i + k] != "#") or s[i - k] == s[i + k] == "#"):
                    k += 1
                arm[i] = k
                k -= 1
                if i + k > right:
                    left = i - k
                    right = i + k
            # s[i-arm[i]+1: i+arm[i]] is palindrome substring for every i
            return arm

        m = ac.read_int()
        mp = ManacherPlindrome()
        mp.manacher = manacher
        cnt = mp.palindrome_length_count(ac.read_str())
        ans = sum(cnt[i] for i in range(2, m + 1, 2))
        ac.st(ans)
        return

    @staticmethod
    def library_check_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/UVA11475
        tag: palindrome_just_end|manacher
        """
        while True:
            s = ac.read_str()
            if not s:
                break
            post = ManacherPlindrome().palindrome_just_end(s)
            if not post:
                ac.st(s + s)
            else:
                x = min(post)
                ac.st(s + s[:x][::-1])
        return

    @staticmethod
    def lc_647(s: str) -> int:
        """
        url: https://leetcode.cn/problems/palindromic-substrings/
        tag: manacher|palindrome_count
        """
        return ManacherPlindrome().palindrome_count(s)
