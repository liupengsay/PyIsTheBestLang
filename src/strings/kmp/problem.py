"""
Algorithm：kmp|find|z-function|circular_section
Description：string|prefix_suffix

====================================LeetCode====================================
28（https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/）kmp|find
214（https://leetcode.cn/problems/shortest-palindrome/）longest_palindrome_prefix
796（https://leetcode.cn/problems/rotate-string/）rotate_string
25（https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/）find|kmp|substring
1392（https://leetcode.cn/problems/longest-happy-prefix/）longest_prefix_suffix|kmp|z_function|template
2223（https://leetcode.cn/problems/sum-of-scores-of-built-strings）z_function
2800（https://leetcode.cn/problems/shortest-string-that-contains-three-strings/）kmp|prefix_suffix|greedy|brain_teaser
2851（https://leetcode.cn/problems/string-transformation/description/）kmp|matrix_fast_power|string_hash
3008（https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/）kmp|find
686（https://leetcode.cn/problems/repeated-string-match/）kmp|find|greedy
2800（https://leetcode.cn/problems/shortest-string-that-contains-three-strings/）
1397（https://leetcode.cn/problems/find-all-good-strings/）
459（https://leetcode.cn/problems/repeated-substring-pattern/）
1163（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）kmp|matrix_dp|kmp_automaton

=====================================LuoGu======================================
P3375（https://www.luogu.com.cn/problem/P3375）longest_prefix_suffix|find
P4391（https://www.luogu.com.cn/problem/P4391）brain_teaser|kmp|n-pi[n-1]
P3435（https://www.luogu.com.cn/problem/P3435）
P4824（https://www.luogu.com.cn/problem/P4824）
P2375（https://www.luogu.com.cn/problem/P2375）
P7114（https://www.luogu.com.cn/problem/P7114）
P3426（https://www.luogu.com.cn/problem/P3426）
P3193（https://www.luogu.com.cn/problem/P3193）
P4503（https://www.luogu.com.cn/problem/P4503）
P3538（https://www.luogu.com.cn/problem/P3538）
P4036（https://www.luogu.com.cn/problem/P4036）
P5410（https://www.luogu.com.cn/problem/P5410）
P1368（https://www.luogu.com.cn/problem/P1368）

===================================CodeForces===================================
1326D2（https://codeforces.com/problemset/problem/1326/D2）manacher|greedy|prefix_suffix|longest_prefix_suffix|palindrome_substring
432D（https://codeforces.com/contest/432/problem/D）kmp|z-function|sorted_list
25E（https://codeforces.com/contest/25/problem/E）kmp|prefix_suffix|greedy|longest_common_prefix_suffix
126B（https://codeforces.com/contest/126/problem/B）kmp|z-function|classical|brute_force
471D（https://codeforces.com/contest/471/problem/D）kmp|brain_teaser|classical|diff_array
346B（https://codeforces.com/contest/346/problem/B）kmp|lcs|matrix_dp
494B（https://codeforces.com/contest/494/problem/B）kmp|linear_dp|prefix_sum
1200E（https://codeforces.com/problemset/problem/1200/E）string_hash|kmp
615C（https://codeforces.com/contest/615/problem/C）kmp|linear_dp|specific_plan
1163D（https://codeforces.com/problemset/problem/1163/D）
526D（https://codeforces.com/contest/526/problem/D）brain_teaser|classical|kmp|circular_section
954I（https://codeforces.com/problemset/problem/954/I）
1003F（https://codeforces.com/problemset/problem/1003/F）
25E（https://codeforces.com/problemset/problem/25/E）
808G（https://codeforces.com/problemset/problem/808/G）
182D（https://codeforces.com/problemset/problem/182/D）
526D（https://codeforces.com/problemset/problem/526/D）
535D（https://codeforces.com/problemset/problem/535/D）
1051E（https://codeforces.com/contest/1051/problem/E）
496B（https://codeforces.com/problemset/problem/496/B）

=====================================AcWing=====================================
143（https://www.acwing.com/problem/content/143/）kmp|circular_section
162（https://www.acwing.com/problem/content/162/）z_function|template
3826（https://www.acwing.com/problem/content/3826/）kmp|z_function


=====================================LibraryChecker=====================================
1（https://www.luogu.com.cn/training/53971）
2（https://loj.ac/p/103）
3（https://acm.hdu.edu.cn/showproblem.php?pid=2087）
4（https://oj.socoding.cn/p/1446）
5（https://www.lanqiao.cn/problems/5132/learning/?contest_id=144）
6（https://www.luogu.com.cn/problem/UVA10298）
7（https://www.luogu.com.cn/problem/UVA11022）
8（https://poj.org/problem?id=2406）
9（https://www.luogu.com.cn/problem/UVA455）
10（https://judge.yosupo.jp/problem/zalgorithm）
11（https://codeforces.com/edu/course/2/lesson/3/3/practice/contest/272263/problem/A）
12（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/A）
13（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/B）
14（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/C）
15（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/D）
16（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/E）
17（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/F）
18（https://atcoder.jp/contests/abc284/tasks/abc284_f）
19（https://poj.org/problem?id=1509）
20（https://codeforces.com/gym/103585/problem/K）

"""
import bisect
import math
from collections import Counter
from itertools import permutations
from typing import List

from src.data_structure.sorted_list.template import SortedList
from src.mathmatics.fast_power.template import MatrixFastPower
from src.strings.kmp.template import KMP
from src.utils.fast_io import FastIO, inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3375(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3375
        tag: longest_prefix_suffix|find
        """
        s1 = ac.read_str()
        s2 = ac.read_str()
        m, n = len(s1), len(s2)
        pi = KMP().prefix_function(s2 + "@" + s1)
        for i in range(n, m + n + 1):
            if pi[i] == n:
                ac.st(i - n + 1 - n)
        ac.lst(pi[:n])
        return

    @staticmethod
    def cf_1326d2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1326/D2
        tag: manacher|greedy|prefix_suffix|longest_prefix_suffix|palindrome_substring
        """
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

            a = KMP().find_longest_palindrome(s[i:j + 1])
            b = KMP().find_longest_palindrome(s[i:j + 1], "suffix")
            ans = s[:i + a] + s[j + 1:] if a > b else s[:i] + s[j - b + 1:]
            ac.st(ans)
        return

    @staticmethod
    def lc_214(s: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-palindrome/
        tag: longest_palindrome_prefix
        """
        k = KMP().find_longest_palindrome(s)
        return s[k:][::-1] + s

    @staticmethod
    def lc_796(s: str, goal: str) -> bool:
        """
        url: https://leetcode.cn/problems/rotate-string/
        tag: rotate_string
        """
        ans = KMP().find(s + s, goal)
        return len(ans) > 0 and len(s) == len(goal)

    @staticmethod
    def lc_28(haystack: str, needle: str) -> int:
        """
        url: https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/
        tag: kmp|find
        """
        ans = KMP().find(haystack, needle)
        return ans[0] if ans else -1

    @staticmethod
    def lc_1392(s: str) -> str:
        """
        url: https://leetcode.cn/problems/longest-happy-prefix/
        tag: longest_prefix_suffix|kmp|z_function|template
        """
        pi = KMP().prefix_function(s)
        return s[:pi[-1]]

    @staticmethod
    def lc_2223(s: str) -> int:
        """
        url: https://leetcode.cn/problems/sum-of-scores-of-built-strings
        tag: z_function
        """
        ans = sum(KMP().z_function(s)) + len(s)
        return ans

    @staticmethod
    def lg_p4391(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4391
        tag: brain_teaser|kmp|n-pi[n-1]|classical
        """

        n = ac.read_int()
        s = ac.read_str()
        pi = KMP().prefix_function(s)
        ac.st(n - pi[-1])
        return

    @staticmethod
    def ac_143(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/143/
        tag: kmp|circular_section
        """
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
                if pi[i] and (i + 1) % (i + 1 - pi[i]) == 0:
                    ac.lst([i + 1, (i + 1) // (i + 1 - pi[i])])
            ac.st("")
        return

    @staticmethod
    def ac_160(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/162/
        tag: z_function|template
        """

        n, m, q = ac.read_list_ints()
        s = ac.read_str()
        t = ac.read_str()
        st = t + "#" + s
        z = KMP().z_function(st)
        cnt = Counter(z[m + 1:])
        for _ in range(q):
            x = ac.read_int()
            ac.st(cnt[x])
        return

    @staticmethod
    def cf_25e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/25/problem/E
        tag: kmp|prefix_suffix|greedy|longest_common_prefix_suffix
        """

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
    def lc_2800(aa: str, bb: str, cc: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-string-that-contains-three-strings/
        tag: kmp|prefix_suffix|greedy|brain_teaser
        """

        def check(a, b):
            c = b + "#" + a
            f = KMP().prefix_function(c)
            m = len(b)
            if max(f[m:]) == m:
                return a
            x = f[-1]
            return a + b[x:]

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
        """
        url: https://leetcode.cn/problems/string-transformation/description/
        tag: kmp|matrix_fast_power|string_hash|brain_teaser
        """

        n = len(s)
        mod = 10 ** 9 + 7
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
    def ac_3826(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3826/
        tag: kmp|z_function
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            z = KMP().z_function(s)
            pre = 0
            for i in range(1, n):
                if z[i] == n - i and pre >= z[i]:
                    ac.st(s[:z[i]])
                    break
                pre = ac.max(pre, z[i])
            else:
                ac.st("not exist")
        return

    @staticmethod
    def lc_3008(s: str, a: str, b: str, k: int) -> List[int]:
        """"
        url: https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/
        tag: kmp|find
        """
        lst1 = KMP().find(s, a)
        lst2 = KMP().find(s, b)
        ans = []
        for i in lst1:
            j = bisect.bisect_left(lst2, i)
            for x in [j - 1, j, j + 1]:
                if 0 <= x < len(lst2) and abs(lst2[x] - i) <= k:
                    ans.append(i)
                    break
        return ans

    @staticmethod
    def cf_1200e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1200/problem/E
        tag: string_hash|kmp
        """
        ac.read_int()
        lst = ac.read_list_strs()
        ans = []
        kmp = KMP()
        for word in lst:
            if not ans:
                ans.extend(list(word))
            else:
                m = len(word)
                k = ac.min(len(ans), m)
                s = list(word[:k]) + ans[-k:]
                z = kmp.z_function(s)
                inter = 0
                for i in range(1, k + 1):
                    if z[-i] == i:
                        inter = i
                for j in range(inter, m):
                    ans.append(word[j])
        ac.st("".join(ans))
        return

    @staticmethod
    def lc_186(a: str, b: str) -> int:
        """
        url: https://leetcode.cn/problems/repeated-string-match/
        tag: kmp|find|greedy
        """
        ceil = len(b) // len(a) + 2
        kmp = KMP()
        ans = kmp.find(ceil * a, b)
        if not ans:
            return -1
        res = (ans[0] + len(b) + len(a) - 1) // len(a)
        return res

    @staticmethod
    def cf_126b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/126/problem/B
        tag: kmp|z-function|classical|brute_force
        """
        s = ac.read_str()
        n = len(s)
        z = KMP().z_function(s)
        pre = 0
        for i in range(1, n):
            if z[i] == n - i and pre >= z[i]:
                ac.st(s[:z[i]])
                break
            pre = ac.max(pre, z[i])
        else:
            ac.st("Just a legend")
        return

    @staticmethod
    def cf_471d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/471/problem/D
        tag: kmp|brain_teaser|classical|diff_array
        """
        m, n = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        if n == 1:
            ac.st(m)
            return
        if m < n:
            ac.st(0)
            return
        a = [a[i + 1] - a[i] for i in range(m - 1)]
        b = [b[i + 1] - b[i] for i in range(n - 1)]
        ans = len(KMP().find_lst(a, b, -10 ** 9 - 1))
        ac.st(ans)
        return

    @staticmethod
    def cf_432d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/432/problem/D
        tag: kmp|z-function|sorted_list
        """
        s = ac.read_str()
        z = KMP().z_function(s)
        lst = SortedList()
        n = len(s)
        ans = []
        for i in range(1, n):
            if z[i] == n - i:
                ans.append((n - i, lst.bisect_right(i - n) + 2))
            lst.add(-z[i])
        ans.reverse()
        ans.append((n, 1))
        ac.st(len(ans))
        for ls in ans:
            ac.lst(ls)
        return

    @staticmethod
    def cf_346b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/346/problem/B
        tag: kmp|lcs|matrix_dp|specific_plan|classical
        """
        s = [ord(w) - ord("A") for w in ac.read_str()]
        t = [ord(w) - ord("A") for w in ac.read_str()]
        virus = [ord(w) - ord("A") for w in ac.read_str()]
        m, n, k = len(s), len(t), len(virus)

        nxt = [[-1] * 26 for _ in range(k)]
        kmp = KMP()
        pre = []
        for i in range(k):
            for j in range(26):
                nxt[i][j] = kmp.prefix_function(virus + [-1] + pre + [j])[-1]
            pre.append(virus[i])

        dp = [[[0] * (k + 1) for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                for x in range(k):
                    a, b = dp[i + 1][j][x], dp[i][j + 1][x]
                    dp[i][j][x] = ac.max(a, b)
                    if s[i] == t[j] and nxt[x][s[i]] < k and dp[i + 1][j + 1][nxt[x][s[i]]] + 1 > dp[i][j][x]:
                        dp[i][j][x] = dp[i + 1][j + 1][nxt[x][s[i]]] + 1
        length = dp[0][0][0]
        if length:
            ans = []
            i, j, x = 0, 0, 0
            while len(ans) < length:
                if dp[i][j][x] == dp[i + 1][j][x]:
                    i += 1
                elif dp[i][j][x] == dp[i][j + 1][x]:
                    j += 1
                else:
                    ans.append(s[i])
                    i, j, x = i + 1, j + 1, nxt[x][s[i]]
            ac.st("".join([chr(i + ord("A")) for i in ans]))
        else:
            ac.st("0")
        return

    @staticmethod
    def cf_494b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/494/problem/B
        tag: kmp|linear_dp|prefix_sum
        """
        s = ac.read_str()
        t = ac.read_str()
        m, n = len(t), len(s)
        pi = KMP().prefix_function(t + "#" + s)
        mod = 10 ** 9 + 7
        dp = [0] * (n + 1)
        pre = [0] * (n + 1)
        dp[0] = pre[0] = 1
        last = -1
        for i in range(1, n + 1):
            if pi[i + m] == m:
                last = i - m + 1
            if last != -1:
                dp[i] = dp[i - 1] + pre[last - 1]
            else:
                dp[i] = dp[i - 1]
            dp[i] %= mod
            pre[i] = (pre[i - 1] + dp[i]) % mod
        ac.st((dp[-1] - 1) % mod)
        return

    @staticmethod
    def cf_615c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/615/problem/C
        tag: kmp|linear_dp|specific_plan
        """
        s = ac.read_str()
        t = ac.read_str()
        m, n = len(s), len(t)
        dp = [inf] * (n + 1)
        dp[0] = 0
        state = [() for _ in range(n + 1)]
        for i in range(n):
            pre = t[:i + 1][::-1]
            z_flip = KMP().z_function(pre + "#" + s)
            for j in range(i + 2, i + 2 + m):
                if z_flip[j] and dp[i + 1 - z_flip[j]] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[i + 1 - z_flip[j]] + 1
                    a, b = j - i - 2, j - i - 2 + z_flip[j] - 1
                    state[i + 1] = (b, a)

            z_flip = KMP().z_function(pre + "#" + s[::-1])
            for j in range(i + 2, i + 2 + m):
                if z_flip[j] and dp[i + 1 - z_flip[j]] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[i + 1 - z_flip[j]] + 1
                    a, b = j - i - 2, j - i - 2 + z_flip[j] - 1
                    state[i + 1] = (m - 1 - b, m - 1 - a)
        if dp[-1] == inf:
            ac.st(-1)
        else:
            ans = []
            x = n
            while x:
                ans.append(state[x])
                x -= abs(state[x][0] - state[x][1]) + 1
            ac.st(len(ans))
            ans.reverse()
            for ls in ans:
                ac.lst((x + 1 for x in ls))
        return

    @staticmethod
    def cf_1163d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1163/problem/D
        tag: kmp|matrix_dp|kmp_automaton
        """
        c = [ord(w) - ord("a") for w in ac.read_str()]
        s = [ord(w) - ord("a") for w in ac.read_str()]
        t = [ord(w) - ord("a") for w in ac.read_str()]
        n, m, k = len(c), len(s), len(t)

        nxt_s = KMP().kmp_automaton(s)
        nxt_t = KMP().kmp_automaton(t)

        dp = [[-inf] * (k + 1) * (m + 1) for _ in range(2)]
        dp[0][0] = 0
        for i in range(n):
            if chr(c[i] + ord("a")) == "*":
                lst = list(range(26))
            else:
                lst = [c[i]]
            for j in range(m + 1):
                for x in range(k + 1):
                    dp[(i & 1) ^ 1][j * (k + 1) + x] = -inf
            for j in range(m + 1):
                for x in range(k + 1):
                    cur = dp[i & 1][j * (k + 1) + x]
                    if cur == -inf:
                        continue
                    for w in lst:
                        tmp = cur
                        jj = nxt_s[j * 26 + w]
                        xx = nxt_t[x * 26 + w]
                        if jj == m:
                            tmp += 1
                        if xx == k:
                            tmp -= 1
                        if tmp > dp[(i & 1) ^ 1][jj * (k + 1) + xx]:
                            dp[(i & 1) ^ 1][jj * (k + 1) + xx] = tmp
        ac.st(max(dp[n & 1]))
        return

    @staticmethod
    def cf_526d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/526/problem/D
        tag: brain_teaser|classical|kmp|circular_section
        """
        n, k = ac.read_list_ints()
        ans = ["0"] * n
        s = ac.read_str()
        pi = KMP().prefix_function(s)
        for i in range(n):
            c = i + 1 - pi[i]
            low = math.ceil((i + 1) / ((k + 1) * c))
            high = (i + 1) // (k * c)
            if low <= high:
                ans[i] = "1"
        ac.st("".join(ans))
        return
