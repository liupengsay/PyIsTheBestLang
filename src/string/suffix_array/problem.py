"""
Algorithm：suffix_array
Description：suffix_array

====================================LeetCode====================================
718（https://leetcode.cn/problems/maximum-length-of-repeated-subarray/）suffix_array|height|sa|lcp|trick|lcs
1754（https://leetcode.cn/problems/largest-merge-of-two-strings/）largest|suffix_array
1698（https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/）suffix_array|height
1044（https://leetcode.cn/problems/longest-duplicate-substring/）suffix_array|height|classical
1062（https://leetcode.cn/problems/longest-repeating-substring/）suffix_array|height|classical
2261（ https://leetcode.cn/problems/k-divisible-elements-subarrays/）suffix_array|height|different_limited_substring
1923（https://leetcode.cn/problems/longest-common-subpath/）suffix_array|lcs|lcp|monotonic_queue
1977（https://leetcode-cn.com/problems/number-of-ways-to-separate-numbers/）
1018（https://leetcode-cn.com/problems/string-matching-in-an-array/）
100508（https://leetcode.cn/problems/find-the-lexicographically-largest-string-from-the-box-i/）suffix_array|greedy|brute_force

=====================================LuoGu======================================
P3809（https://www.luogu.com.cn/problem/P3809）suffix_array
P2852（https://www.luogu.com.cn/problem/P2852）binary_search|suffix_array|height|monotonic_queue|string_hash
P2852（https://www.luogu.com.cn/problem/P2408）suffix_array|height
P3804（https://www.luogu.com.cn/problem/P3804）suffix_array|height|monotonic_stack
P4248（https://www.luogu.com.cn/problem/P4248）suffix_array|height|lcp|monotonic_stack
P3975（https://www.luogu.com.cn/problem/P3975）greedy|bfs|suffix_array|height
P3796（https://www.luogu.com.cn/problem/P3796）suffix_array|height|sa|monotonic_stack|prefix_sum
P5546（https://www.luogu.com.cn/problem/P5546）suffix_array|lcs|lcp|monotonic_queue
P4341（https://www.luogu.com.cn/problem/P4341）suffix_array|height
P4070（https://www.luogu.com.cn/problem/P4070）
P6095（https://www.luogu.com.cn/problem/P6095）
P2870（https://www.luogu.com.cn/problem/P2870）suffix_array|greedy|implemention|classical

=====================================AcWing=====================================
142（https://www.acwing.com/problem/content/142/）suffix_array|template
140（https://www.acwing.com/problem/content/140/）suffix_array|lexicographical_order|lcp|sparse_table|sub_string|classical

=====================================CodeForces=====================================
123D（https://codeforces.com/problemset/problem/123/D）suffix_array|height|monotonic_stack
271D（https://codeforces.com/contest/271/problem/D）suffix_array|height|different_limited_substring
802I（https://codeforces.com/contest/802/problem/I）suffix_array|height|monotonic_stack
128B（https://codeforces.com/contest/128/problem/B）greedy|bfs|suffix_array|height
427D（https://codeforces.com/contest/427/problem/D）suffix_array|height|sa|lcp|trick|lcs
1526E（https://codeforces.com/contest/1526/problem/E）suffix_array|reverse_thinking|comb|construction
611D（https://codeforces.com/problemset/problem/611/D）
600A（https://codeforces.com/problemset/problem/600/A）
873F（https://codeforces.com/contest/873/problem/F）suffix_array|reverse_thinking|lcp|prefix_sum

=====================================AtCoder=====================================
ABC141E（https://atcoder.jp/contests/abc141/tasks/abc141_e）suffix_array|height|binary_search|string_hash
ABC213F（https://atcoder.jp/contests/abc213/tasks/abc213_f）suffix_array|height|lcp
ABC272F（https://atcoder.jp/contests/abc272/tasks/abc272_f）suffix_array|sa|trick

=====================================LibraryChecker=====================================
1（https://judge.yosupo.jp/problem/suffixarray）suffix_array
2（https://judge.yosupo.jp/problem/number_of_substrings）suffix_array|sa
3（https://www.hackerrank.com/challenges/morgan-and-a-string/）lexicographical_order|classical
4（https://loj.ac/p/111）suffix_array|template
5（https://atcoder.jp/contests/practice2/tasks/practice2_i）suffix_array|height
6（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/A）suffix_array|height
7（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/H）suffix_array|height
8（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/F）suffix_array|height|lcp|brute_force
9（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/D）suffix_array|height|monotonic_stack
10（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/B）suffix_array|height|sa|lcp|trick|lcs
11（https://www.spoj.com/problems/LCS2/）suffix_array|lcs|lcp|monotonic_queue
12（https://loj.ac/p/171）suffix_array|lcs|lcp|monotonic_queue
13（https://www.spoj.com/problems/PHRASES/）suffix_array|height|sa|lcp|binary_search
14（https://poj.org/problem?id=3581)
15（https://poj.org/problem?id=1226）
16（https://poj.org/problem?id=3450）
17（https://poj.org/problem?id=3729）
18（https://onlinejudge.u-aizu.ac.jp/problems/2292）
19（https://poj.org/problem?id=3415）
20（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/E）suffix_array|monotonic_stack|height|counter
21（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/C）suffix_array|lexicographical_order|lcp|sparse_table|sub_string|classical
22（https://codeforces.com/edu/course/2/lesson/2/2/practice/contest/269103/problem/A）suffix_array
23（https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/A）suffix_array|height|lcp|monotonic_stack|prefix_sum|binary_search
24（https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/B）suffix_array|height|lcp|monotonic_stack|prefix_sum|binary_search
25（https://codeforces.com/edu/course/2/lesson/2/4/practice/contest/269119/problem/A）suffix_array|height|lcp


1（https://www.codechef.com/problems/CABABAA）sparse_table|suffix_array|monotonic_stack
"""
import math
from collections import deque
from functools import cmp_to_key
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.math.comb_perm.template import Combinatorics
from src.string.suffix_array.template import SuffixArray
from src.struct.monotonic_stack.template import Rectangle
from src.struct.sorted_list.template import SortedList
from src.struct.sparse_table.template import SparseTable
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1754_1(word1: str, word2: str) -> str:
        """
        url: https://leetcode.cn/problems/largest-merge-of-two-strings/
        tag: largest|suffix_array
        """

        s = [ord(w) - ord("a") + 1 for w in word1] + [0] + [ord(w) - ord("a") + 1 for w in word2]
        sa, rk, height = SuffixArray().build(s, 27)
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
        """
        url: https://leetcode.cn/problems/largest-range_merge_to_disjoint-of-two-strings/
        tag: largest|suffix_array
        """
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
    def lg_p3809(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3809
        tag: suffix_array
        """
        words = ([str(x) for x in range(10)]
                 + [chr(i + ord("A")) for i in range(26)]
                 + [chr(i + ord("a")) for i in range(26)])
        ind = {st: i for i, st in enumerate(words)}
        rk = [ind[w] for w in ac.read_str()]
        sa = SuffixArray().build(rk, len(ind))[0]
        ac.lst([x + 1 for x in sa])
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/suffixarray
        tag: suffix_array
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa = SuffixArray().build(s, 26)[0]
        ac.lst(sa)
        return

    @staticmethod
    def library_check_2(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/number_of_substrings
        tag: suffix_array|sa
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        n = len(s)
        ans = sum(height)
        ac.st(n * (n + 1) // 2 - ans)
        return

    @staticmethod
    def ac_142(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/142/
        tag: suffix_array|template
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        ac.lst(sa)
        ac.lst(height)
        return

    @staticmethod
    def library_check_3(ac=FastIO()):
        """
        url: https://www.hackerrank.com/challenges/morgan-and-a-string/
        tag: lexicographical_order|classical
        """
        for _ in range(ac.read_int()):
            word1 = ac.read_str()
            word2 = ac.read_str()

            s = [ord(w) - ord("A") for w in word1] + [26] + [ord(w) - ord("A") for w in word2] + [26]
            sa, rk, height = SuffixArray().build(s, 27)
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
            ac.st(ans)
        return

    @staticmethod
    def lc_1698(s: str) -> int:
        """
        url: https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/
        tag: suffix_array|height
        """
        s = [ord(w) - ord("a") for w in s]
        sa, rk, height = SuffixArray().build(s, 26)
        n = len(s)
        ans = sum(height)
        return n * (n + 1) // 2 - ans

    @staticmethod
    def library_check_4(ac=FastIO()):
        """
        url: https://loj.ac/p/111
        tag: suffix_array|template
        """
        words = ([str(x) for x in range(10)]  # TLE
                 + [chr(i + ord("A")) for i in range(26)]
                 + [chr(i + ord("a")) for i in range(26)])
        ind = {st: i for i, st in enumerate(words)}
        rk = [ind[w] for w in ac.read_str()]
        sa = SuffixArray().build(rk, len(ind))[0]
        ac.lst([x + 1 for x in sa])
        return

    @staticmethod
    def lc_1044(s: str) -> str:
        """
        url: https://leetcode.cn/problems/longest-duplicate-substring/
        tag: suffix_array|height|classical
        """
        sa, rk, height = SuffixArray().build([ord(w) - ord("a") for w in s], 26)
        j = height.index(max(height))
        i = sa[j]
        return s[i: i + height[j]]

    @staticmethod
    def lc_1062(s: str) -> int:
        """
        url: https://leetcode.cn/problems/longest-repeating-substring/
        tag: suffix_array|height|classical
        """
        sa, rk, height = SuffixArray().build([ord(w) - ord("a") for w in s], 26)
        return max(height)

    @staticmethod
    def abc_141e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc141/tasks/abc141_e
        tag: suffix_array|height|binary_search|string_hash
        """
        n = ac.read_int()
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)

        def check(x):
            lst = [sa[0]]
            for i in range(1, n):
                if height[i] >= x:
                    lst.append(sa[i])
                else:
                    a, b = min(lst), max(lst)
                    if a + x <= b:
                        return True
                    lst = [sa[i]]
            a, b = min(lst), max(lst)
            if a + x <= b:
                return True
            return False

        ans = BinarySearch().find_int_right(0, max(height), check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2852(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2852
        tag: binary_search|suffix_array|height|monotonic_queue
        """
        n, k = ac.read_list_ints()
        s = [ac.read_int() for _ in range(n)]
        ind = {num: i for i, num in enumerate(sorted(list(set(s))))}
        s = [ind[d] for d in s]

        _, _, height = SuffixArray().build(s, len(ind))
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - (k - 1):
                stack.popleft()
            while stack and stack[-1][0] >= height[i]:
                stack.pop()
            stack.append((height[i], i))
            if i >= (k - 1) - 1:
                ans.append(stack[0][0])
        ac.st(max(ans))
        return

    @staticmethod
    def lg_p2408(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2408
        tag: suffix_array|height
        """
        ac.read_int()
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        n = len(s)
        ans = sum(height)
        ac.st(n * (n + 1) // 2 - ans)
        return

    @staticmethod
    def library_check_5(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/practice2/tasks/practice2_i
        tag: suffix_array|height
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        n = len(s)
        ans = sum(height)
        ac.st(n * (n + 1) // 2 - ans)
        return

    @staticmethod
    def lc_2261(nums: List[int], k: int, p: int) -> int:
        """
        url: https://leetcode.cn/problems/k-divisible-elements-subarrays/
        tag: suffix_array|height|different_limited_substring
        """
        cnt = res = 0
        n = len(nums)
        j = 0
        right = [0] * n
        for i in range(n):
            while j < n and cnt + (nums[j] % p == 0) <= k:
                cnt += (nums[j] % p == 0)
                j += 1
            res += j - i
            right[i] = j
            cnt -= (nums[i] % p == 0)
        sa, rk, height = SuffixArray().build(nums, max(nums) + 1)
        dup = sum(min(height[i], right[sa[i]] - sa[i]) for i in range(1, n))
        res -= dup
        return res

    @staticmethod
    def library_check_6(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/A
        tag: suffix_array|height
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        n = len(s)
        ans = sum(height)
        ac.st(n * (n + 1) // 2 - ans)
        return

    @staticmethod
    def library_check_7(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/H
        tag: suffix_array|height
        """
        words = ([chr(i + ord("A")) for i in range(26)]
                 + [chr(i + ord("a")) for i in range(26)])
        ind = {st: i for i, st in enumerate(words)}
        s = [ind[w] for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, len(ind))
        n = len(s)
        ans = n * (n + 1) * (n + 2) // 6 - sum(height[i] * (height[i] + 1) // 2 for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def cf_271d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/271/problem/D
        tag: suffix_array|height|different_limited_substring
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        good = [1 - int(w) for w in ac.read_str()]
        k = ac.read_int()
        n = len(s)
        j = cnt = ans = 0
        right = [0] * n
        for i in range(n):
            while j < n and cnt + good[s[j]] <= k:
                cnt += good[s[j]]
                j += 1
            ans += j - i
            right[i] = j
            cnt -= good[s[i]]
        sa, rk, height = SuffixArray().build(s, 26)
        dup = sum(min(height[i], right[sa[i]] - sa[i]) for i in range(1, n))
        ans -= dup
        ac.st(ans)
        return

    @staticmethod
    def library_check_8(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/F
        tag: suffix_array|height|lcp|brute_force
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        n = len(s)
        sa, rk, height = SuffixArray().build(s, 26)
        ans = 1
        st = SparseTable(height, min)

        def lcp(ii, jj):
            ri, rj = rk[ii], rk[jj]
            if ri > rj:
                ri, rj = rj, ri
            if ri == rj:
                return n - sa[ri]
            return st.query(ri + 1, rj)

        for x in range(1, n):
            for i in range(0, n - x, x):
                rep_len = lcp(i, i + x)
                rep_cnt = rep_len // x + 1
                p = i - (x - rep_len % x)
                if p >= 0 and lcp(p, p + x) >= x:
                    rep_cnt += 1
                if rep_cnt > ans:
                    ans = rep_cnt
        ac.st(ans)
        return

    @staticmethod
    def cf_123d_1(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/123/D
        tag: suffix_array|height|monotonic_stack
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        ans = Rectangle().compute_number([h + 1 for h in height])
        ac.st(ans)
        return

    @staticmethod
    def cf_123d_2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/123/D
        tag: suffix_array|height|monotonic_stack
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        n = len(s)
        sa, rk, height = SuffixArray().build(s, 26)
        ans = Rectangle().compute_number(height)
        ans += sum(n - sa[j] for j in range(n))
        ac.st(ans)
        return

    @staticmethod
    def library_check_9(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/D
        tag: suffix_array|height|monotonic_stack
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        n = len(s)
        sa, rk, height = SuffixArray().build(s, 26)
        ans = Rectangle().compute_number([h for h in height])
        ans += n * (n + 1) // 2
        ac.st(ans)
        return

    @staticmethod
    def cf_802i(ac=FastIO()):
        """
        url: https://codeforces.com/contest/802/problem/I
        tag: suffix_array|height|monotonic_stack
        """

        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            sa, rk, height = SuffixArray().build([ord(w) - ord("a") for w in s], 26)
            stack = [[-1, 0]]
            ans = n * (n + 1) // 2
            for i in range(n):
                while stack[-1][0] >= 0 and height[i] < height[stack[-1][0]]:
                    stack.pop()
                stack.append([i, stack[-1][1] + (i - stack[-1][0]) * height[i]])
                ans += stack[-1][1] * 2
            ac.st(ans)

        return

    @staticmethod
    def lg_p3804(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3804
        tag: suffix_array|height|monotonic_stack
        """

        s = ac.read_str()
        sa, rk, height = SuffixArray().build([ord(w) - ord("a") for w in s], 26)
        n = len(s)
        left = [1] * n
        right = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and height[stack[-1]] > height[i]:
                right[stack.pop()] = i - 1
            stack.append(i)

        stack = []
        for i in range(n - 1, -1, -1):
            while stack and height[stack[-1]] > height[i]:
                left[stack.pop()] = i + 1
            stack.append(i)

        ans = 0
        for i in range(n):
            cur = height[i] * (right[i] - left[i] + 2)
            ans = ans if ans > cur else cur

        ac.st(ans)
        return

    @staticmethod
    def abc_213f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc213/tasks/abc213_f
        tag: suffix_array|height|lcp
        """
        n = ac.read_int()
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, nums = SuffixArray().build(s, 26)

        pre = [0] * n  # initial can be 0 or -1 dependent on usage
        stack = [0]
        for i in range(1, n):  # can be also range(n-1, -1, -1) dependent on usage
            while stack and nums[stack[-1]] > nums[i]:  # can be < or > or <=  or >=  dependent on usage
                stack.pop()  # can be i or i-1 dependent on usage
            pre[i] = pre[stack[-1]] + nums[i] * (i - stack[-1])
            stack.append(i)

        nums.append(0)
        post = [0] * (n + 1)  # initial can be 0 or -1 dependent on usage
        stack = [n]
        for i in range(n - 1, -1, -1):  # can be also range(n-1, -1, -1) dependent on usage
            while stack and nums[stack[-1]] > nums[i]:  # can be < or > or <=  or >=  dependent on usage
                stack.pop()  # can be i or i-1 dependent on usage
            post[i] = post[stack[-1]] + nums[i] * (stack[-1] - i)
            stack.append(i)

        ans = [n - i for i in range(n)]
        for i in range(n):
            ans[i] += pre[rk[i]] + post[rk[i] + 1]
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p4248(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4248
        tag: suffix_array|height|lcp
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(s, 26)
        height.append(0)
        n = len(s)
        pre = [0] * (n + 1)
        stack = [0]
        for i in range(n):
            if i:
                pre[i] = pre[i - 1]
            while stack[-1] and height[stack[-1]] > height[i]:
                j = stack.pop()
                pre[i] -= (j - stack[-1]) * height[j]
            pre[i] += (i - stack[-1]) * height[i]
            stack.append(i)

        post = [0] * (n + 1)
        stack = [n]
        for i in range(n - 1, -1, -1):
            post[i] = post[i + 1]
            while stack[-1] < n and height[stack[-1]] > height[i]:
                j = stack.pop()
                post[i] -= (stack[-1] - j) * height[j]
            post[i] += (stack[-1] - i) * height[i]
            stack.append(i)

        suf = ans = 0
        for i in range(n - 1, -1, -1):
            ans += (n - i) * (n - i - 1) + suf
            suf += n - i

        for i in range(n):
            ans -= pre[rk[i]] + post[rk[i] + 1]
        ac.st(ans)
        return

    @staticmethod
    def cf_128b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/128/problem/B
        tag: greedy|bfs|suffix_array|height
        """
        s = ac.read_str()
        lst = [ord(w) - ord("a") for w in s]
        k = ac.read_int()
        n = len(s)
        if k > n * (n + 1) // 2:
            ac.st("No such line.")
            return
        ans = []
        ind = list(range(n))
        while ind and k > 0:
            dct = [[] for _ in range(26)]
            for i in ind:
                dct[lst[i]].append(i)
            for x in range(26):
                cur = sum(n - i for i in dct[x])
                if cur < k:
                    k -= cur
                else:
                    ans.append(chr(x + ord("a")))
                    ind = [i + 1 for i in dct[x] if i + 1 < n]
                    k -= len(dct[x])
                    break
        ac.st("".join(ans))
        return

    @staticmethod
    def lc_718(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-length-of-repeated-subarray/
        tag: suffix_array|height|sa|lcp|trick|lcs
        """
        m, n = len(nums1), len(nums2)
        nums = nums1 + [101] + nums2
        sa, rk, height = SuffixArray().build(nums, 102)
        ans = 0
        for i in range(1, m + n + 1):
            if (sa[i - 1] < m < sa[i] or sa[i - 1] > m > sa[i]) and height[i] > ans:
                ans = height[i]
        return ans

    @staticmethod
    def library_check_10(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/B
        tag: suffix_array|height|sa|lcp|trick|lcs
        """
        s = ac.read_str()
        nums1 = [ord(w) - ord("a") for w in s]
        nums2 = [ord(w) - ord("a") for w in ac.read_str()]
        m, n = len(nums1), len(nums2)
        nums = nums1 + [26] + nums2
        sa, rk, height = SuffixArray().build(nums, 27)
        ans = 0
        ind = -1
        for i in range(1, m + n + 1):
            if (sa[i - 1] < m < sa[i] or sa[i - 1] > m > sa[i]) and height[i] > ans:
                ans = height[i]
                ind = min(sa[i - 1], sa[i])
        ac.st(s[ind:ind + ans])
        return

    @staticmethod
    def cf_427d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/427/problem/D
        tag: suffix_array|height|sa|lcp|trick|lcs
        """
        s = ac.read_str()
        nums1 = [ord(w) - ord("a") for w in s]
        nums2 = [ord(w) - ord("a") for w in ac.read_str()]
        m, n = len(nums1), len(nums2)
        nums = nums1 + [26] + nums2
        sa, rk, height = SuffixArray().build(nums, 27)
        height.append(0)
        ans = math.inf
        for i in range(1, m + n + 1):
            if not height[i]:
                continue
            if sa[i - 1] < m < sa[i] or sa[i - 1] > m > sa[i]:
                a = max(height[i - 1], height[i + 1])
                if a + 1 <= height[i] and a + 1 < ans:
                    ans = a + 1
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def abc_272f_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc272/tasks/abc272_f
        tag: suffix_array|sa|trick
        """
        n = ac.read_int()
        nums1 = [ord(w) - ord("a") + 1 for w in ac.read_str()]
        nums2 = [ord(w) - ord("a") + 1 for w in ac.read_str()]
        nums = nums1 + nums1 + [0] + nums2 + nums2 + [27]
        sa, rk, height = SuffixArray().build(nums, 28)
        ans = post = 0
        for i in range(4 * n + 1, -1, -1):
            if 2 * n + 1 < sa[i] < 3 * n + 2:
                post += 1
            elif sa[i] < n:
                ans += post
        ac.st(ans)
        return

    @staticmethod
    def abc_272f_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc272/tasks/abc272_f
        tag: suffix_array|sa|trick
        """
        n = ac.read_int()
        s = [ord(w) - ord("a") for w in ac.read_str()]
        t = [ord(w) - ord("a") for w in ac.read_str()]
        lst = s + s + t + t
        sa, rk, height = SuffixArray().build(lst, 26)
        ans = 0
        ind = SortedList(list(range(2 * n, 3 * n)))
        j = 0
        for i in range(4 * n):
            if height[i] < n:
                while j < i:
                    if 2 * n <= sa[j] < 3 * n:
                        ind.discard(sa[j])
                    j += 1
            if 0 <= sa[i] < n:
                ans += len(ind)
        ac.st(ans)
        return

    @staticmethod
    def library_check_11(ac=FastIO()):
        """
        url: https://www.spoj.com/problems/LCS2/
        tag: suffix_array|lcs|lcp|monotonic_queue
        """

        lst = []  # TLE
        while True:
            s = ac.read_str()
            if not s:
                break
            lst.append([ord(w) - ord("a") for w in s])

        ind = []
        nums = []
        k = len(lst)
        for i in range(k):
            m = len(lst[i])
            ind.extend([i] * m)
            ind.append(i)
            nums.extend(lst[i])
            nums.append(26 + i)
        if k == 1:
            return len(nums)
        sa, rk, height = SuffixArray().build(nums, 26 + k)

        def add(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                if not item[w]:
                    cnt += 1
                item[w] += 1
            return

        def remove(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                item[w] -= 1
                if not item[w]:
                    cnt -= 1
            return

        ans = cnt = 0
        j = 1
        item = [0] * k
        n = len(height)
        stack = deque()
        for i in range(1, n):
            while stack and stack[0] < i:
                stack.popleft()
            while j < n and cnt < k:
                add(j)
                while stack and height[stack[-1]] > height[j]:
                    stack.pop()
                stack.append(j)
                j += 1
            if cnt == k and stack and height[stack[0]] > ans:
                ans = height[stack[0]]
            remove(i)

        ac.st(ans)
        return

    @staticmethod
    def library_check_12(ac=FastIO()):
        """
        url: https://loj.ac/p/171
        tag: suffix_array|lcs|lcp|monotonic_queue
        """

        lst = []
        for _ in range(ac.read_int()):
            lst.append([ord(w) - ord("a") for w in ac.read_str()])

        ind = []
        nums = []
        k = len(lst)
        for i in range(k):
            m = len(lst[i])
            ind.extend([i] * m)
            ind.append(i)
            nums.extend(lst[i])
            nums.append(26 + i)
        if k == 1:
            return len(nums)
        sa, rk, height = SuffixArray().build(nums, 26 + k)

        def add(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                if not item[w]:
                    cnt += 1
                item[w] += 1
            return

        def remove(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                item[w] -= 1
                if not item[w]:
                    cnt -= 1
            return

        ans = cnt = 0
        j = 1
        item = [0] * k
        n = len(height)
        stack = deque()
        for i in range(1, n):
            while stack and stack[0] < i:
                stack.popleft()
            while j < n and cnt < k:
                add(j)
                while stack and height[stack[-1]] > height[j]:
                    stack.pop()
                stack.append(j)
                j += 1
            if cnt == k and stack and height[stack[0]] > ans:
                ans = height[stack[0]]
            remove(i)

        ac.st(ans)
        return

    @staticmethod
    def lc_1923(n: int, paths: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/longest-common-subpath/
        tag: suffix_array|lcs|lcp|monotonic_queue
        """
        ind = []
        nums = []
        k = len(paths)
        for i in range(k):
            m = len(paths[i])
            ind.extend([i] * m)
            ind.append(i)
            nums.extend(paths[i])
            nums.append(n + i + 1)
        if k == 1:
            return len(nums)
        sa, rk, height = SuffixArray().build(nums, n + k + 1)

        def add(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                if not item[w]:
                    cnt += 1
                item[w] += 1
            return

        def remove(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                item[w] -= 1
                if not item[w]:
                    cnt -= 1
            return

        ans = cnt = 0
        j = 1
        item = [0] * k
        n = len(height)
        stack = deque()
        for i in range(1, n):
            while stack and stack[0] < i:
                stack.popleft()
            while j < n and cnt < k:
                add(j)
                while stack and height[stack[-1]] > height[j]:
                    stack.pop()
                stack.append(j)
                j += 1
            if cnt == k and stack and height[stack[0]] > ans:
                ans = height[stack[0]]
            remove(i)

        return ans

    @staticmethod
    def library_check_13(ac=FastIO()):
        """
        url: https://www.spoj.com/problems/PHRASES/
        tag: suffix_array|height|sa|lcp|binary_search
        """

        for _ in range(ac.read_int()):
            k = ac.read_int()
            lst = []
            for _ in range(k):
                lst.append([ord(w) - ord("a") for w in ac.read_str()])

            ind = []
            nums = []
            k = len(lst)
            for i in range(k):
                m = len(lst[i])
                ind.extend([i] * m)
                ind.append(i)
                nums.extend(lst[i])
                nums.append(26 + i)

            sa, rk, height = SuffixArray().build(nums, 26 + k)
            n = len(height)

            def check(x):

                index = [[] for _ in range(k)]
                for ii in range(n):
                    if height[ii] >= x:
                        a, b = sa[ii - 1], sa[ii]
                        index[ind[a]].append(a)
                        index[ind[b]].append(b)
                    else:
                        if all(len(ls) > 0 for ls in index) and all(max(ls) - min(ls) >= x for ls in index):
                            return True
                        index = [[] for _ in range(k)]

                if all(len(ls) > 0 for ls in index) and all(max(ls) - min(ls) >= x for ls in index):
                    return True
                return False

            ans = BinarySearch().find_int_right(0, min(len(ls) for ls in lst) // 2, check)
            ac.st(ans)
        return

    @staticmethod
    def lg_p3796(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3796
        tag: suffix_array|height|sa|monotonic_stack|prefix_sum
        """
        while True:  # MLE
            k = ac.read_int()
            if not k:
                break
            lst = []
            for _ in range(k + 1):
                lst.append([ord(w) - ord("a") for w in ac.read_str()])

            ind = []
            nums = []
            for i in range(k + 1):
                m = len(lst[i])
                ind.extend([i] * m)
                ind.append(i)
                nums.extend(lst[i])
                nums.append(26 + i)

            sa, _, height = SuffixArray().build(nums, 26 + k + 1)
            del nums

            n = len(height)
            height.append(0)

            right = [n - 1] * (n + 1)
            stack = []
            for i in range(n):
                while stack and height[stack[-1]] > height[i]:
                    right[stack.pop()] = i - 1
                stack.append(i)

            left = [0] * (n + 1)
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and height[stack[-1]] > height[i]:
                    left[stack.pop()] = i + 1
                stack.append(i)
            pre = ac.accumulate([int(ind[sa[i]] == k) for i in range(n)])
            cnt = [0] * k
            for i in range(n):
                j = sa[i]
                if ind[j] < k and height[i] == len(lst[ind[j]]):
                    a, b = left[i], right[i]

                    if pre[b + 1] - pre[a - 1] > cnt[ind[j]]:
                        cnt[ind[j]] = pre[b + 1] - pre[a - 1]
            ceil = max(cnt)
            ac.st(ceil)
            for i in range(k):
                if cnt[i] == ceil:
                    ac.st("".join([chr(ord("a") + w) for w in lst[i]]))
        return

    @staticmethod
    def cf_1526e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1526/problem/E
        tag: suffix_array|reverse_thinking|comb|construction
        """
        mod = 998244353
        n, k = ac.read_list_ints()
        sa = ac.read_list_ints()
        cb = Combinatorics(n + k, mod)
        rk = [0] * n
        for i in range(n):
            rk[sa[i]] = i
        m = 0
        for i in range(1, n):
            if sa[i] < n - 1 and sa[i - 1] < n - 1 and rk[sa[i] + 1] > rk[sa[i - 1] + 1]:
                m += 1
            elif sa[i - 1] == n - 1:
                m += 1
        ans = cb.comb(m + k, n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5546(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5546
        tag: suffix_array|lcs|lcp|monotonic_queue
        """
        lst = []
        for _ in range(ac.read_int()):
            lst.append([ord(w) - ord("a") for w in ac.read_str()])

        ind = []
        nums = []
        k = len(lst)
        for i in range(k):
            m = len(lst[i])
            ind.extend([i] * m)
            ind.append(i)
            nums.extend(lst[i])
            nums.append(26 + i)
        if k == 1:
            return len(nums)
        sa, rk, height = SuffixArray().build(nums, 26 + k)

        def add(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                if not item[w]:
                    cnt += 1
                item[w] += 1
            return

        def remove(ii):
            nonlocal cnt
            for w in [ind[sa[ii - 1]], ind[sa[ii]]]:
                item[w] -= 1
                if not item[w]:
                    cnt -= 1
            return

        ans = cnt = 0
        j = 1
        item = [0] * k
        n = len(height)
        stack = deque()
        for i in range(1, n):
            while stack and stack[0] < i:
                stack.popleft()
            while j < n and cnt < k:
                add(j)
                while stack and height[stack[-1]] > height[j]:
                    stack.pop()
                stack.append(j)
                j += 1
            if cnt == k and stack and height[stack[0]] > ans:
                ans = height[stack[0]]
            remove(i)

        ac.st(ans)
        return

    @staticmethod
    def lg_p4341(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4341
        tag: suffix_array|height
        """
        n = ac.read_int()
        lst = [int(w) for w in ac.read_str()]
        sa, rk, height = SuffixArray().build(lst, 2)

        for i in range(n):
            j = sa[i]
            for x in range(height[i] + 1, n - j + 1):
                y = 1
                for k in range(i + 1, n):
                    if height[k] >= x:
                        y += 1
                    else:
                        break
                if y > 1:
                    ac.st(y)
        return

    @staticmethod
    def cf_837f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/873/problem/F
        tag: suffix_array|reverse_thinking|lcp|prefix_sum
        """
        n = ac.read_int()
        a = ac.read_str()[::-1]
        s = ac.read_str()[::-1]
        sa, rk, height = SuffixArray().build([ord(w) - ord("a") for w in a], 26)

        left = [1] * n
        right = [n - 1] * n
        stack = []
        for i in range(1, n):
            while stack and height[stack[-1]] > height[i]:
                right[stack.pop()] = i - 1
            stack.append(i)

        stack = []
        for i in range(n - 1, -1, -1):
            while stack and height[stack[-1]] > height[i]:
                left[stack.pop()] = i + 1
            stack.append(i)

        pre = ac.accumulate([1 - int(s[i]) for i in sa])
        ans = max(height[i] * (pre[right[i] + 1] - pre[left[i] - 1]) for i in range(n))
        # special case
        for i in sa:
            if s[i] == "0" and n - i > ans:
                ans = n - i
        ac.st(ans)
        return

    @staticmethod
    def library_check_21(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/C
        tag: suffix_array|lexicographical_order|lcp|sparse_table|sub_string|classical
        """
        s = [ord(w) - 33 for w in ac.read_str()]
        n = len(s)
        sa, rk, height = SuffixArray().build(s, 95)
        st = SparseTable(height, min)
        q = ac.read_int()
        queries = [tuple(ac.read_list_ints_minus_one()) for _ in range(q)]

        def lcp(ii, jj):
            ri, rj = rk[ii], rk[jj]
            if ri > rj:
                ri, rj = rj, ri
            if ri == rj:
                return n - sa[ri]
            return st.query(ri + 1, rj)

        def compare(a, b):
            i1, j1 = a
            i2, j2 = b
            x = lcp(i1, i2)
            x = min(x, j1 - i1 + 1)
            x = min(x, j2 - i2 + 1)
            if x == j1 - i1 + 1:
                if x == j2 - i2 + 1:
                    return -1 if i1 < i2 else 1
                return -1
            if x == j2 - i2 + 1:
                return 1
            return -1 if s[i1 + x] < s[i2 + x] else 1

        queries.sort(key=cmp_to_key(compare))
        for q in queries:
            ac.lst([x + 1 for x in q])
        return

    @staticmethod
    def ac_140(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/140/
        tag: suffix_array|lexicographical_order|lcp|sparse_table|sub_string|classical
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        n = len(s)
        sa, rk, height = SuffixArray().build(s, 26)
        st = SparseTable(height, min)

        def lcp(ii, jj):
            ri, rj = rk[ii], rk[jj]
            if ri > rj:
                ri, rj = rj, ri
            if ri == rj:
                return n - sa[ri]
            return st.query(ri + 1, rj)

        for _ in range(ac.read_int()):
            i1, j1, i2, j2 = ac.read_list_ints_minus_one()
            x = lcp(i1, i2)
            x = min(x, j1 - i1 + 1)
            if j1 - i1 + 1 == j2 - i2 + 1 == x:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def library_check_20(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/E
        tag: suffix_array|monotonic_stack|height|counter
        """
        n, m = ac.read_list_ints()
        s = ac.read_list_ints_minus_one()
        sa, rk, height = SuffixArray().build(s, m)

        left = [1] * n
        right = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and height[stack[-1]] > height[i]:
                right[stack.pop()] = i - 1
            stack.append(i)

        stack = []
        for i in range(n - 1, -1, -1):
            while stack and height[stack[-1]] > height[i]:
                left[stack.pop()] = i + 1
            stack.append(i)

        ans = n
        res = [0, n]
        for i in range(n):
            cur = height[i] * (right[i] - left[i] + 2)
            if cur > ans:
                ans = cur
                res = [sa[i], sa[i] + height[i]]
        ac.st(ans)
        ac.st(res[1] - res[0])
        ac.lst([x + 1 for x in s[res[0]:res[1]]])
        return

    @staticmethod
    def library_check_22(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/2/practice/contest/269103/problem/A
        tag: suffix_array
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        sa, _, _ = SuffixArray().build(s, 26)
        ac.lst([len(sa)] + sa)
        return

    @staticmethod
    def library_check_23_1(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/A
        tag: suffix_array|height|lcp|monotonic_stack|prefix_sum
        """
        t = ac.read_str()
        words = [ac.read_str() for _ in range(ac.read_int())]

        lst = [ord(w) - ord("a") for w in t] + [26]
        ind = []
        m = len(lst)
        length = dict()
        k = len(words)
        for i, s in enumerate(words):
            ind.append(len(lst))
            cur = [ord(w) - ord("a") for w in s] + [26 + i + 1]
            length[len(lst)] = len(cur) - 1
            lst.extend(cur)
        sa, rk, height = SuffixArray().build(lst, 26 + k + 1)
        n = len(sa)

        right = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and height[stack[-1]] > height[i]:
                right[stack.pop()] = i - 1
            stack.append(i)

        left = [1] * n
        stack = []
        for i in range(n - 1, 0, -1):
            while stack and height[stack[-1]] > height[i]:
                left[stack.pop()] = i + 1
            stack.append(i)

        pre = ac.accumulate([int(i < m - 1) for i in sa])
        ans = []
        for i in ind:
            j = rk[i]
            if height[j] >= length[i]:
                ll = left[j]
                if pre[j + 1] > pre[ll - 1]:
                    ans.append(True)
                    continue
            if j + 1 < n and height[j + 1] >= length[i]:
                rr = right[j + 1]
                if pre[rr + 1] > pre[j]:
                    ans.append(True)
                    continue
            ans.append(False)
        for a in ans:
            ac.st("Yes" if a else "No")
        return

    @staticmethod
    def library_check_23_2(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/A
        tag: suffix_array|height|lcp|monotonic_stack|prefix_sum|binary_search
        """

        t = ac.read_str()
        lst = [ord(w) - ord("a") for w in t]
        sa, _, _ = SuffixArray().build(lst, 26)
        m = len(t)

        def check(i):
            j = sa[i]
            return t[j:j + n] >= s

        bs = BinarySearch()
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            x = bs.find_int_left(0, m - 1, check)
            if t[sa[x]:sa[x] + n] == s:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def library_check_24_1(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/B
        tag: suffix_array|height|lcp|monotonic_stack|prefix_sum|binary_search
        """

        t = ac.read_str()
        words = [ac.read_str() for _ in range(ac.read_int())]

        lst = [ord(w) - ord("a") for w in t] + [26]
        ind = []
        m = len(lst)
        length = dict()
        k = len(words)
        for i, s in enumerate(words):
            ind.append(len(lst))
            cur = [ord(w) - ord("a") for w in s] + [26 + i + 1]
            length[len(lst)] = len(cur) - 1
            lst.extend(cur)
        sa, rk, height = SuffixArray().build(lst, 26 + k + 1)
        n = len(sa)

        right = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and height[stack[-1]] > height[i]:
                right[stack.pop()] = i - 1
            stack.append(i)

        left = [1] * n
        stack = []
        for i in range(n - 1, 0, -1):
            while stack and height[stack[-1]] > height[i]:
                left[stack.pop()] = i + 1
            stack.append(i)

        pre = ac.accumulate([int(i < m - 1) for i in sa])
        for i in ind:
            ans = 0
            j = rk[i]
            if height[j] >= length[i]:
                ll = left[j]
                ans += pre[j + 1] - pre[ll - 1]
            if j + 1 < n and height[j + 1] >= length[i]:
                rr = right[j + 1]
                ans += pre[rr + 1] - pre[j + 1]
            ac.st(ans)
        return

    @staticmethod
    def library_check_24_2(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/3/practice/contest/269118/problem/B
        tag: suffix_array|height|lcp|monotonic_stack|prefix_sum|binary_search
        """

        t = ac.read_str()
        lst = [ord(w) - ord("a") for w in t]
        sa, _, _ = SuffixArray().build(lst, 26)
        m = len(t)

        def check(i):
            j = sa[i]
            return t[j:j + n] >= s

        def check2(i):
            j = sa[i]
            return t[j:j + n + 1] <= s

        bs = BinarySearch()
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)

            x = bs.find_int_left(0, m - 1, check)
            if t[sa[x]:sa[x] + n] == s:
                s += chr(1 + ord("z"))
                y = bs.find_int_right(0, m - 1, check2)
                ac.st(y - x + 1)
            else:
                ac.st(0)
        return

    @staticmethod
    def library_check_25(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/2/4/practice/contest/269119/problem/A
        tag: suffix_array|height|lcp
        """

        t = ac.read_str()

        lst = [ord(w) - ord("a") for w in t]

        sa, rk, height = SuffixArray().build(lst, 26)
        n = len(sa)
        ac.lst([n] + sa)
        ac.lst(height)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/CABABAA
        tag: sparse_table|suffix_array|monotonic_stack
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = [ord(w) - ord("a") for w in ac.read_str()]
            sa, rk, height = SuffixArray().build(s, 26)
            st = SparseTable(height, min)
            sa.reverse()

            def lcp(ii, jj):
                ri, rj = rk[ii], rk[jj]
                if ri > rj:
                    ri, rj = rj, ri
                if ri == rj:
                    return n - sa[ri]
                return st.query(ri + 1, rj)

            def compare(a, b):
                i1, j1 = a
                i2, j2 = b
                x = lcp(i1, i2)
                x = min(x, j1 - i1 + 1)
                x = min(x, j2 - i2 + 1)
                if x == j1 - i1 + 1:
                    if x == j2 - i2 + 1:
                        return -1 if i1 < i2 else 1
                    return -1
                if x == j2 - i2 + 1:
                    return 1
                return -1 if s[i1 + x] < s[i2 + x] else 1

            stack = [n]
            for i in range(n - 1, -1, -1):
                while len(stack) >= 2 and i < stack[-1] and compare((stack[-1], stack[-2] - 1),
                                                                    (i, stack[-2] - 1)) == -1:
                    stack.pop()
                stack.append(i)
            ans = []
            right = n - 1
            for i in stack[1:]:
                ans.extend(s[i:right + 1])
                right = i - 1
            ac.st("".join(chr(x + ord("a")) for x in ans))
        return

    @staticmethod
    def lg_p2870(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2870
        tag: suffix_array|greedy|implemention|classical
        """
        s = []
        for _ in range(ac.read_int()):
            s.append(ord(ac.read_str()[0]) - ord("A"))
        n = len(s)
        s += s[::-1]

        _, rk, _ = SuffixArray().build(s, 26)

        ans = []
        ll, rr = 0, n - 1
        for _ in range(n):
            if rk[ll] < rk[2 * n - 1 - rr]:
                ans.append(s[ll])
                ll += 1
            else:
                ans.append(s[rr])
                rr -= 1
            if len(ans) == 80:
                res = "".join(chr(x + ord("A")) for x in ans)
                ac.st(res)
                ans = []
        if ans:
            res = "".join(chr(x + ord("A")) for x in ans)
            ac.st(res)
        return

    @staticmethod
    def lc_100508(word: str, m: int) -> str:
        """
        url: https://leetcode.cn/problems/find-the-lexicographically-largest-string-from-the-box-i/
        tag: suffix_array|greedy|brute_force
        """
        if m == 1:
            return word

        n = len(word)
        ans = (0, 0)
        s = [ord(w) - ord("a") for w in word]
        sa, rk, height = SuffixArray().build(s, 26)
        st = SparseTable(height, min)

        def lcp(ii, jj):
            ri, rj = rk[ii], rk[jj]
            if ri > rj:
                ri, rj = rj, ri
            if ri == rj:
                return n - sa[ri]
            return st.query(ri + 1, rj)

        def compare(a, b):
            i1, j1 = a
            i2, j2 = b
            x = lcp(i1, i2)
            x = min(x, j1 - i1 + 1)
            x = min(x, j2 - i2 + 1)
            if x == j1 - i1 + 1:
                if x == j2 - i2 + 1:
                    return -1 if i1 < i2 else 1
                return -1
            if x == j2 - i2 + 1:
                return 1
            return -1 if s[i1 + x] < s[i2 + x] else 1

        for i in range(n):
            need = max(0, m - i - 1)
            cur = (i, n - need - 1)
            if compare(ans, cur) == -1:
                ans = cur
        return word[ans[0]:ans[1] + 1]
