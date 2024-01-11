"""
Algorithm：suffix_array
Description：suffix_array

====================================LeetCode====================================
1754（https://leetcode.cn/problems/largest-merge-of-two-strings/）largest|suffix_array
1698（https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/）suffix_array|height
1044（https://leetcode.cn/problems/longest-duplicate-substring/）suffix_array|height|classical
1062（https://leetcode.cn/problems/longest-repeating-substring/）suffix_array|height|classical
2261（ https://leetcode.cn/problems/k-divisible-elements-subarrays/）suffix_array|height

=====================================LuoGu======================================
P3809（https://www.luogu.com.cn/problem/P3809）suffix_array
P2852（https://www.luogu.com.cn/problem/P2852）binary_search|suffix_array|height|monotonic_queue|string_hash
P2852（https://www.luogu.com.cn/problem/P2408）suffix_array|height

=====================================AcWing=====================================
140（https://www.acwing.com/problem/content/142/）suffix_array|template

=====================================AtCoder=====================================
ABC141E（https://atcoder.jp/contests/abc141/tasks/abc141_e）suffix_array|height|binary_search|string_hash

=====================================LibraryChecker=====================================
1（https://judge.yosupo.jp/problem/suffixarray）suffix_array
2（https://judge.yosupo.jp/problem/number_of_substrings）suffix_array|sa
3（https://www.hackerrank.com/challenges/morgan-and-a-string/）smallest|lexicographical_order|classical
4（https://loj.ac/p/111）suffix_array|template
5（https://atcoder.jp/contests/practice2/tasks/practice2_i）suffix_array|height
6（https://codeforces.com/edu/course/2/lesson/2/5/practice/contest/269656/problem/A）suffix_array|height
7（https://codeforces.com/edu/course/2/lesson/3/4/practice/contest/272262/problem/H）suffix_array|height


"""

from collections import deque
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.strings.suffix_array.template import SuffixArray
from src.utils.fast_io import FastIO


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
    def ac_140(ac=FastIO()):
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
        tag: smallest|lexicographical_order|classical
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
        tag: suffix_array|height
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
