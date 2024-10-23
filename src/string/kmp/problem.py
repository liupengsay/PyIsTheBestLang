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
2800（https://leetcode.cn/problems/shortest-string-that-contains-three-strings/）kmp|prefix_suffix|greed|brain_teaser
2851（https://leetcode.cn/problems/string-transformation/description/）kmp|matrix_fast_power|string_hash
3008（https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/）kmp|find
686（https://leetcode.cn/problems/repeated-string-match/）kmp|find|greed
1397（https://leetcode.cn/problems/find-all-good-strings/）digital_dp|kmp_automaton
459（https://leetcode.cn/problems/repeated-substring-pattern/）kmp|circular_section
1163（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）kmp|matrix_dp|kmp_automaton
3292（https://leetcode.cn/problems/minimum-number-of-valid-strings-to-form-target-ii/）kmp|greed|linear_dp
100433（https://leetcode.com/problems/find-the-occurrence-of-first-almost-equal-substring/）z_function|greed|classical

=====================================LuoGu======================================
P3375（https://www.luogu.com.cn/problem/P3375）longest_prefix_suffix|find
P4391（https://www.luogu.com.cn/problem/P4391）brain_teaser|kmp|n-pi[n-1]
P3435（https://www.luogu.com.cn/problem/P3435）kmp|longest_circular_section|prefix_function_reverse|classical
P4824（https://www.luogu.com.cn/problem/P4824）
P2375（https://www.luogu.com.cn/problem/P2375）kmp|z-function|diff_array
P7114（https://www.luogu.com.cn/problem/P7114）
P3426（https://www.luogu.com.cn/problem/P3426）
P3193（https://www.luogu.com.cn/problem/P3193）kmp_automaton|matrix_fast_power|matrix_dp
P4036（https://www.luogu.com.cn/problem/P4036）kmp|z-function
P5410（https://www.luogu.com.cn/problem/P5410）kmp|z-function
P1368（https://www.luogu.com.cn/problem/P1368）
P3121（https://www.luogu.com.cn/problem/P3121）
P5829（https://www.luogu.com.cn/problem/P5829）kmp|z-function|fail_tree|classical|border|longest_common_border|tree_lca
P8112（https://www.luogu.com.cn/problem/P8112）z_function|point_set|range_min|classical


===================================CodeForces===================================
1326D2（https://codeforces.com/problemset/problem/1326/D2）manacher|greed|prefix_suffix|longest_prefix_suffix|palindrome_substring
432D（https://codeforces.com/contest/432/problem/D）kmp|z-function|sorted_list
25E（https://codeforces.com/contest/25/problem/E）kmp|prefix_suffix|greed|longest_common_prefix_suffix
126B（https://codeforces.com/contest/126/problem/B）kmp|z-function|classical|brute_force
471D（https://codeforces.com/contest/471/problem/D）kmp|brain_teaser|classical|diff_array
346B（https://codeforces.com/contest/346/problem/B）kmp|lcs|matrix_dp
494B（https://codeforces.com/contest/494/problem/B）kmp|linear_dp|prefix_sum
1200E（https://codeforces.com/problemset/problem/1200/E）string_hash|kmp
615C（https://codeforces.com/contest/615/problem/C）kmp|linear_dp|specific_plan
1163D（https://codeforces.com/problemset/problem/1163/D）kmp|matrix_dp|kmp_automaton
526D（https://codeforces.com/contest/526/problem/D）brain_teaser|classical|kmp|circular_section
954I（https://codeforces.com/problemset/problem/954/I）
808G（https://codeforces.com/contest/808/problem/G）kmp|kmp_automaton|z-function|matrix_dp
182D（https://codeforces.com/problemset/problem/182/D）kmp|circular_section|num_factor
535D（https://codeforces.com/problemset/problem/535/D）kmp|z-function|union_find
1051E（https://codeforces.com/contest/1051/problem/E）kmp|z-function|linear_dp
1015F（https://codeforces.com/contest/1015/problem/F）kmp_automaton|matrix_dp
1690F（https://codeforces.com/contest/1690/problem/F）permutation_circle|kmp|circle_section
1968G2（https://codeforces.com/contest/1968/problem/G2）z_algorithm|offline_query|binary_search|brute_force|preprocess
1984D（https://codeforces.com/contest/1984/problem/D）kmp|z_function|euler_series|brain_teaser|brute_force

=====================================AcWing=====================================
143（https://www.acwing.com/problem/content/143/）kmp|circular_section
162（https://www.acwing.com/problem/content/162/）z_function|template
3826（https://www.acwing.com/problem/content/3826/）kmp|z_function

=====================================AtCoder=====================================
ABC284F（https://atcoder.jp/contests/abc284/tasks/abc284_f）
ABC343G（https://atcoder.jp/contests/abc343/tasks/abc343_g）kmp|state_dp|classical
ABC257G（https://atcoder.jp/contests/abc257/tasks/abc257_g）z_function|point_set|range_min|classical

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
10（https://judge.yosupo.jp/problem/zalgorithm）kmp|z-function
11（https://codeforces.com/edu/course/2/lesson/3/3/practice/contest/272263/problem/A）kmp|z-function
12（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/A）kmp|circular_section
13（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/B）kmp|find|z-function
14（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/C）kmp|diff_array|z-function
15（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/D）kmp|find_longest_palindrome
16（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/E）kmp|z-function
17（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/F）kmp|z-function
18（https://poj.org/problem?id=1509）
19（https://codeforces.com/gym/103585/problem/K）


1（https://www.codechef.com/problems/BREAKSTRING）kmp|z_function

"""
import bisect
import math
from collections import Counter
from functools import lru_cache
from itertools import permutations
from typing import List

from src.graph.union_find.template import UnionFind
from src.math.fast_power.template import MatrixFastPower
from src.math.number_theory.template import NumFactor
from src.string.kmp.template import KMP
from src.structure.segment_tree.template import PointSetRangeMin
from src.structure.sorted_list.template import SortedList
from src.tree.tree_dp.template import WeightedTree
from src.util.fast_io import FastIO


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
        tag: manacher|greed|prefix_suffix|longest_prefix_suffix|palindrome_substring
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
        tag: kmp|prefix_suffix|greed|longest_common_prefix_suffix
        """

        s = [ac.read_str() for _ in range(3)]

        ind = list(range(3))
        ans = sum(len(w) for w in s)
        kmp = KMP()
        for item in permutations(ind, 3):
            cur = len(kmp.merge_b_from_a(kmp.merge_b_from_a(s[item[0]], s[item[1]]), s[item[2]]))
            if cur < ans:
                ans = cur
        ac.st(ans)
        return

    @staticmethod
    def lc_2800(a: str, b: str, c: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-string-that-contains-three-strings/
        tag: kmp|prefix_suffix|greed|brain_teaser
        """

        s = [a, b, c]
        ind = list(range(3))
        ans = sum(len(w) for w in s)
        kmp = KMP()
        res = a + b + c
        for item in permutations(ind, 3):
            st = kmp.merge_b_from_a(kmp.merge_b_from_a(s[item[0]], s[item[1]]), s[item[2]])
            cur = len(st)
            if cur < ans or (cur == ans and st < res):
                ans = cur
                res = st
        return res

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
                pre = max(pre, z[i])
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
                k = min(len(ans), m)
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
        tag: kmp|find|greed
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
            pre = max(pre, z[i])
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
                    dp[i][j][x] = max(a, b)
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
        dp = [math.inf] * (n + 1)
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
        if dp[-1] == math.inf:
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

        dp = [[-math.inf] * (k + 1) * (m + 1) for _ in range(2)]
        dp[0][0] = 0
        for i in range(n):
            if chr(c[i] + ord("a")) == "*":
                lst = list(range(26))
            else:
                lst = [c[i]]
            for j in range(m + 1):
                for x in range(k + 1):
                    dp[(i & 1) ^ 1][j * (k + 1) + x] = -math.inf
            for j in range(m + 1):
                for x in range(k + 1):
                    cur = dp[i & 1][j * (k + 1) + x]
                    if cur == -math.inf:
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

    @staticmethod
    def cf_808g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/808/problem/G
        tag: kmp|kmp_automaton|z-function|matrix_dp
        """
        s = ac.read_str()
        t = ac.read_str()
        m, n = len(s), len(t)
        z = KMP().z_function(t)
        ind = [0]
        for i in range(1, n):
            if z[i] == n - i:
                ind.append(n - i)

        dp = [[-math.inf] * n for _ in range(2)]
        pre = 0
        dp[pre][0] = 0
        for w in s:
            cur = 1 - pre
            for j in range(n):
                dp[cur][j] = -math.inf
            dp[cur][0] = max(dp[pre])
            for j in range(n):
                if t[j] == w or w == "?":
                    if j == n - 1:
                        for x in ind:
                            dp[cur][x] = max(dp[cur][x], dp[pre][j] + 1)
                    else:
                        dp[cur][j + 1] = max(dp[cur][j + 1], dp[pre][j])
            pre = cur
        ac.st(max(dp[pre]))
        return

    @staticmethod
    def lc_1397(n: int, s1: str, s2: str, evil: str) -> int:

        """
        url: https://leetcode.cn/problems/find-all-good-strings/
        tag: digital_dp|kmp_automaton
        """

        @lru_cache(None)
        def dfs(i, is_floor_limit, is_ceil_limit, sub_evil):
            if sub_evil == m:
                return 0
            if i == n:
                return 1
            res = 0
            start = 0 if not is_floor_limit else s1[i]
            end = 25 if not is_ceil_limit else s2[i]
            for w in range(start, end + 1):
                cur_evil = sub_evil + 1 if evil[sub_evil] == w else nxt[sub_evil * 26 + w]
                res += dfs(i + 1, is_floor_limit and s1[i] == w, is_ceil_limit and s2[i] == w, cur_evil)
                res %= mod
            return res

        mod = 10 ** 9 + 7
        m = len(evil)
        s1 = [ord(w) - ord("a") for w in s1]
        s2 = [ord(w) - ord("a") for w in s2]
        evil = [ord(w) - ord("a") for w in evil]
        n = len(s1)
        nxt = KMP().kmp_automaton(evil)
        ans = dfs(0, True, True, 0)
        return ans

    @staticmethod
    def lg_p3435(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3435
        tag: kmp|longest_circular_section|prefix_function_reverse|classical
        """
        n = ac.read_int()
        s = ac.read_str()
        nxt = KMP().prefix_function_reverse(s)
        ans = sum(i + 1 - nxt[i] for i in range(1, n) if nxt[i])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2375(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2375
        tag: kmp|z-function|diff_array
        """
        mod = 10 ** 9 + 7
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            z = KMP().z_function(s)
            diff = [0] * n
            for i in range(1, n):
                if z[i]:
                    x = min(z[i], i)
                    diff[i] += 1
                    if i + x < n:
                        diff[i + x] -= 1
            ans = 1
            for i in range(1, n):
                diff[i] += diff[i - 1]
                ans *= (diff[i] + 1)
                ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p3193(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3193
        tag: kmp_automaton|matrix_fast_power|matrix_dp
        """
        n, m, k = ac.read_list_ints()
        lst = [int(w) for w in ac.read_str()]
        nxt = KMP().kmp_automaton(lst, 10)
        grid = [[0] * (m + 9) for _ in range(m + 9)]
        for i in range(10):
            for j in range(m):
                ind = i if j == 0 else j + 9
                for x in range(10):
                    y = nxt[j * 10 + x]
                    if y == 0:
                        grid[x][ind] = 1
                    elif y < m:
                        grid[y + 9][ind] = 1

        initial = [0] * (m + 9)
        for x in range(10):
            if x == lst[0] and m > 1:
                initial[10] = 1
            elif x != lst[0]:
                initial[x] = 1
        mat = MatrixFastPower().matrix_pow(grid, n - 1, k)
        ans = 0
        for i in range(m + 9):
            ans += sum(mat[i][j] * initial[j] for j in range(m + 9))
        ac.st(ans % k)
        return

    @staticmethod
    def lg_p4036(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4036
        tag: kmp|z-function
        """
        lst = [ord(w) - ord("a") for w in ac.read_str()]
        for _ in range(ac.read_int()):
            cur = ac.read_list_strs()
            if cur[0] == "Q":
                x, y = [int(w) - 1 for w in cur[1:]]
                n = len(lst)
                ans = KMP().z_function(lst[x:] + [-1] + lst[y:])[n - x + 1]
                ac.st(ans)
            elif cur[0] == "R":
                x = int(cur[1]) - 1
                w = ord(cur[2]) - ord("a")
                lst[x] = w
            else:
                x = int(cur[1])
                w = ord(cur[2]) - ord("a")
                lst.insert(x, w)
        return

    @staticmethod
    def lg_p5829(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5829
        tag: kmp|z-function|fail_tree|classical|border|longest_common_border|tree_lca
        """
        lst = [ord(w) - ord("a") for w in ac.read_str()]
        n = len(lst)
        pi = [0] + KMP().prefix_function(lst)

        graph = WeightedTree(n + 1)
        for i in range(1, n + 1):
            if pi[i]:
                graph.add_directed_edge(pi[i], i)
            else:
                graph.add_directed_edge(0, i)
        graph.lca_build_with_multiplication()
        for _ in range(ac.read_int()):
            p, q = ac.read_list_ints()
            ans = graph.lca_get_lca_between_nodes(pi[p], pi[q])
            ac.st(ans)
        return

    @staticmethod
    def cf_182d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/182/problem/D
        tag: kmp|circular_section|num_factor
        """
        s = ac.read_str()
        t = ac.read_str()
        m, n = len(s), len(t)
        c1 = m - KMP().prefix_function(s)[-1]
        c2 = n - KMP().prefix_function(t)[-1]
        if m % c1:
            c1 = m
        if n % c2:
            c2 = n
        if c1 != c2 or s[:c1] != t[:c2]:
            ac.st(0)
        else:
            ac.st(len(NumFactor().get_all_factor(math.gcd(m // c1, n // c2))))
        return

    @staticmethod
    def lc_459(s: str) -> bool:
        """
        url: https://leetcode.cn/problems/repeated-substring-pattern/
        tag: kmp|circular_section
        """
        n = len(s)
        c = n - KMP().prefix_function(s)[-1]
        return n % c == 0 and n // c > 1

    @staticmethod
    def library_check_10(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/zalgorithm
        tag: kmp|z-function
        """
        s = ac.read_str()
        z = KMP().z_function(s)
        z[0] = len(s)
        ac.lst(z)
        return

    @staticmethod
    def library_check_11(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/3/practice/contest/272263/problem/A
        tag: kmp|z-function
        """
        s = ac.read_str()
        z = KMP().z_function(s)
        ac.lst(z)
        return

    @staticmethod
    def lg_p5410(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5410
        tag: kmp|z-function
        """
        a = ac.read_str()
        b = ac.read_str()  # TLE
        m, n = len(a), len(b)
        z = KMP().z_function(b + "#" + a)
        z[0] = n
        ans = 0
        for i in range(n):
            ans ^= (i + 1) * (z[i] + 1)
        ac.st(ans)

        ans = 0
        for i in range(n + 1, n + m + 1):
            ans ^= (i - n) * (z[i] + 1)
        ac.st(ans)
        return

    @staticmethod
    def cf_535d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/535/problem/D
        tag: kmp|z-function|union_find
        """
        n, k = ac.read_list_ints()
        s = ac.read_str()
        m = len(s)
        lst = [""] * n
        uf = UnionFind(n + 1)
        pos = ac.read_list_ints_minus_one()
        for i in pos:
            start = i
            end = i + m - 1
            while uf.find(i) <= end:
                j = uf.find(i)
                lst[j] = s[j - start]
                uf.union_right(j, j + 1)
                i = j + 1

        z = KMP().z_function(list(s) + ["#"] + lst)
        if not all(z[m + 1 + i] == m for i in pos):
            ac.st(0)
            return
        mod = 1000000007
        ans = pow(26, lst.count(""), mod)
        ac.st(ans)
        return

    @staticmethod
    def cf_1051e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1051/problem/E
        tag: kmp|z-function|linear_dp
        """
        s = ac.read_str()
        ll = ac.read_str()
        rr = ac.read_str()
        n = len(s)
        nll = len(ll)
        nrr = len(rr)
        zll = KMP().z_function(ll + "#" + s)[nll + 1:]
        zrr = KMP().z_function(rr + "#" + s)[nrr + 1:]

        def compare_ll(ind):
            lcp = zll[ind]
            if lcp == nll:
                return True
            return s[ind + lcp] >= ll[lcp]

        def compare_rr(ind):
            lcp = zrr[ind]
            if lcp == nrr:
                return True
            return s[ind + lcp] <= rr[lcp]

        mod = 998244353
        dp = [0] * (n + 1)
        post = [0] * (n + 1)
        dp[n] = post[n] = 1
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                if ll == "0":
                    dp[i] = dp[i + 1]
                post[i] = (post[i + 1] + dp[i]) % mod
                continue
            left = i + nll
            right = i + nrr
            if i + nll > n or not compare_ll(i):
                left += 1
            if i + nrr > n or not compare_rr(i):
                right -= 1
            if left <= right and left <= n:
                dp[i] = (post[left] - (post[right + 1] if right < n else 0)) % mod
            post[i] = (post[i + 1] + dp[i]) % mod
        ac.st(dp[0])
        return

    @staticmethod
    def library_check_12(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/A
        tag: kmp|circular_section
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            ans = n - KMP().prefix_function(s)[-1]
            ac.st(s[:ans])
        return

    @staticmethod
    def library_check_13(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/B
        tag: kmp|find|z-function
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            t = ac.read_str()
            ind = KMP().find(s + s, t)
            if not ind:
                ac.st(-1)
            else:
                ac.st(ind[0])
        return

    @staticmethod
    def library_check_14(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/C
        tag: kmp|diff_array|z-function
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            diff = [0] * (n + 1)
            z = KMP().z_function(s + "#" + s)[n + 1:]
            for x in z:
                diff[0] += 1
                if x + 1 <= n:
                    diff[x + 1] -= 1
            for i in range(1, n + 1):
                diff[i] += diff[i - 1]
            ac.lst(diff[1:])
        return

    @staticmethod
    def library_check_15(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/D
        tag: kmp|find_longest_palindrome
        """
        s = ac.read_str()
        ac.st(KMP().find_longest_palindrome(s, "prefix"))
        return

    @staticmethod
    def library_check_16(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/E
        tag: kmp|z-function
        """
        s = ac.read_str()
        t = ac.read_str()

        n = len(s)
        if len(t) != n:
            ac.no()
            return
        if s == t:
            ac.yes()
            ac.st(0)
            return

        z1 = KMP().z_function(t + "#" + s)[n + 1:]
        z2 = KMP().z_function(t[::-1] + "#" + s)[n + 1:]
        for i in range(n):
            if z1[i] == n - i and z2[0] >= i:
                ac.yes()
                ac.st(i)
                return
        ac.no()
        return

    @staticmethod
    def library_check_17(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/F
        tag: kmp|z-function
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            t = ac.read_str()
            if t in s:
                ac.st(s)
                continue
            if s in t:
                ac.st(t)
                continue
            m, n = len(s), len(t)
            z = KMP().z_function(s + "#" + t)[m + 1:]
            ans = s + t
            for i in range(n):
                if z[i] == n - i:
                    cur = t[:-z[i]] + s
                    if len(cur) < len(ans):
                        ans = cur
                    break

            z = KMP().z_function(t + "#" + s)[n + 1:]
            for i in range(m):
                if z[i] == m - i:
                    cur = s[:-z[i]] + t
                    if len(cur) < len(ans):
                        ans = cur
                    break
            ac.st(ans)
        return

    @staticmethod
    def abc_284f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc284/tasks/abc284_f
        tag: kmp|z-function
        """
        n = ac.read_int()
        s = ac.read_str()
        sr = s[::-1]
        z1 = KMP().z_function(sr + s)[2 * n:]  # 4*n
        z2 = KMP().z_function(s + sr)[2 * n:]  # 4*n
        for i in range(n):
            right = 2 * n - n - i
            left = n - right
            if z2[i] >= right and z1[right] >= left:
                ac.st(sr[i:i + n])
                ac.st(right)
                return
        ac.st(-1)
        return

    @staticmethod
    def cf_1015f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1015/problem/F
        tag: kmp_automaton|matrix_dp
        """
        n = ac.read_int()
        s = ac.read_str()
        lst = [int(w == ")") for w in s]
        nxt = KMP().kmp_automaton(lst, 2)
        m = len(s)
        mod = 10 ** 9 + 7

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for _ in range(2 * n):
            ndp = [[0] * (m + 1) for _ in range(n + 1)]
            for s in range(n + 1):
                for p in range(m + 1):
                    if dp[s][p]:
                        for x in [0, 1]:
                            nxt_p = nxt[p * 2 + x] if p != m else m
                            nxt_s = s + 1 if not x else s - 1
                            if n >= nxt_s >= 0:
                                ndp[nxt_s][nxt_p] += dp[s][p]
            dp = [[x % mod for x in ls] for ls in ndp]
        ac.st(dp[0][m])
        return

    @staticmethod
    def cf_1690f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1690/problem/F
        tag: permutation_circle|kmp|circle_section
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = ac.read_str()
            ind = ac.read_list_ints_minus_one()
            dct = {w: i for i, w in enumerate(ind)}
            ans = 1
            visit = [0] * n
            for i in range(n):
                if not visit[i]:
                    lst = [i]
                    visit[i] = 1
                    while not visit[dct[lst[-1]]]:
                        lst.append(dct[lst[-1]])
                        visit[lst[-1]] = 1
                    tmp = [s[j] for j in lst]
                    x = KMP().prefix_function(tmp)[-1]
                    if len(tmp) % (len(tmp) - x) == 0:
                        x = len(tmp) - x
                    else:
                        x = len(tmp)
                    ans = ans * x // math.gcd(ans, x)
            ac.st(ans)
        return

    @staticmethod
    def abc_257g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc257/tasks/abc257_g
        tag: z_function|point_set|range_min|classical
        """
        s = ac.read_str()
        t = ac.read_str()
        n = len(s)
        m = len(t)
        z = KMP().z_function(s + "#" + t)
        tree = PointSetRangeMin(m + 1)
        tree.point_set(m, 0)
        for i in range(m - 1, -1, -1):
            s = z[i + n + 1]
            if s:
                nex = tree.range_min(i + 1, i + s) + 1
                tree.point_set(i, nex)
        ans = tree.range_min(0, 0)
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p8112(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8112
        tag: z_function|point_set|range_min|classical|monotonic_queue|binary_search
        """
        n, m = ac.read_list_ints()  # TLE
        s = ac.read_str()
        t = ac.read_str()
        z = KMP().z_function(s + "#" + t)
        tree = PointSetRangeMin(m + 1)
        tree.point_set(m, 0)
        for i in range(m - 1, -1, -1):
            s = z[i + n + 1]
            if s:
                nex = tree.range_min(i + 1, i + s) + 1
                tree.point_set(i, nex)
        ans = tree.range_min(0, 0)
        ac.st(ans if ans < math.inf else "Fake")
        return

    @staticmethod
    def cf_1968g2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1968/problem/G2
        tag: z_algorithm|offline_query|binary_search|brute_force|preprocess
        """
        for _ in range(ac.read_int()):
            n, ll, rr = ac.read_list_ints()
            s = ac.read_str()
            z = KMP().z_function(s + "#" + s)

            lst = [(z[i + n + 1], i) for i in range(n)]
            lst.sort(reverse=True)

            ans = [-n - 1] * (n + 1)
            i = 0
            ind = SortedList()
            for x in range(n, 0, -1):
                while i < n and lst[i][0] >= x:
                    ind.add(lst[i][1])
                    i += 1
                if not ind:
                    continue
                cur = ind[0]
                cnt = 1
                while cur + x <= ind[-1]:
                    cur = ind[ind.bisect_left(cur + x)]
                    cnt += 1
                ans[x] = -cnt

            res = []
            for x in range(ll, rr + 1):
                res.append(bisect.bisect_right(ans, -x) - 1)

            ac.lst(res)
        return

    @staticmethod
    def cf_1984d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1984/problem/D
        tag: kmp|z_function|euler_series|brain_teaser|brute_force
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            post = [-1] * (n + 1)
            start = -1
            for i in range(n - 1, -1, -1):
                if s[i] != "a":
                    start = i
                post[i] = start
            if post[0] == -1:
                ac.st(n - 1)
                continue
            z = KMP().z_function(s[start:])
            ans = 0
            for j in range(start, n):
                pre = start
                length = j - start + 1
                k = j + 1
                flag = 1
                while k < n:
                    cur = post[k] - k
                    k = post[k]
                    if k == -1:
                        break
                    pre = min(pre, cur)
                    if z[k - start] >= length:
                        k += length
                    else:
                        flag = 0
                        break
                if flag:
                    ans += pre + 1
            ac.st(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/BREAKSTRING
        tag: kmp|z_function
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            if n % 2:
                ac.st(0)
                continue
            z1 = KMP().z_function(s)
            z2 = KMP().z_function(s[::-1])
            ans = 0
            for i in range(n // 2 + 1):
                p = i
                r = n // 2 - i
                if z1[i] >= p and z2[r] >= r:
                    ans += 1
            ac.st(ans)
        return

    @staticmethod
    def lc_3292(words: List[str], target: str) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-valid-strings-to-form-target-ii/
        tag: kmp|greed|linear_dp
        """
        m = len(target)
        post = [0] * m
        k = len(words)
        for i in range(k):
            s = words[i] + target
            tmp = len(words[i])
            z = KMP().z_function(s)
            for j in range(tmp, tmp + m):
                if z[j]:
                    post[j - tmp] = max(post[j - tmp], min(z[j], tmp))

        ans = 0
        right = -1
        nex = -1
        for i, num in enumerate(post):
            if right < i:
                if nex < i - 1:
                    return -1
                ans += 1
                nex = max(nex, i + num - 1)
                right = nex
            else:
                nex = max(nex, i + num - 1)
        return ans if right >= m - 1 else -1

    @staticmethod
    def lc_100433(s: str, pattern: str) -> int:
        """
        url: https://leetcode.com/problems/find-the-occurrence-of-first-almost-equal-substring/
        tag: z_function|greed|classical
        """
        m, n = len(s), len(pattern)
        z1 = KMP().z_function(pattern + s)
        z2 = KMP().z_function((s + pattern)[::-1])[::-1]
        for i in range(m - n + 1):
            left = min(z1[n + i], n)
            if left < n - 1:
                if z2[i + n - 1] >= n - 1 - left:
                    return i
            else:
                return i
        return -1
