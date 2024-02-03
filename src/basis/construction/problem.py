"""

Algorithm：construction
Description：greedy|sort|construction|specific_plan

====================================LeetCode====================================
280（https://leetcode.cn/problems/wiggle-sort/）construction|sort|odd_even
2663（https://leetcode.cn/problems/lexicographically-smallest-beautiful-string/）greedy|construction|palindrome_substring|lexicographical_order
1982（https://leetcode.cn/problems/find-array-given-subset-sums/）construction
1253（https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/）construction|greedy|brain_teaser
2573（https://leetcode.cn/problems/find-the-string-with-lcp/）lcp|construction|union_find

=====================================LuoGu======================================
P8846（https://www.luogu.com.cn/problem/P8846）greedy|construction
P2902（https://www.luogu.com.cn/problem/P2902）construction
P5823（https://www.luogu.com.cn/problem/P5823）construction
P7383（https://www.luogu.com.cn/problem/P7383）greedy|construction
P7947（https://www.luogu.com.cn/problem/P7947）greedy|construction|product_n_sum_k|prime_factorization
P9101（https://www.luogu.com.cn/problem/P9101）construction|directed_graph|no_circe
P8976（https://www.luogu.com.cn/problem/P8976）brute_force|construction
P8910（https://www.luogu.com.cn/problem/P8910）permutation_circle|construction
P8880（https://www.luogu.com.cn/problem/P8880）brain_teaser|construction|odd_even

===================================CodeForces===================================
1396A（https://codeforces.com/problemset/problem/1396/A）greedy|construction
1118E（https://codeforces.com/problemset/problem/1118/E）implemention|greedy|construction
960C（https://codeforces.com/problemset/problem/960/C）greedy|construction
1793B（https://codeforces.com/contest/1793/problem/B）brain_teaser|greedy|construction
1375D（https://codeforces.com/problemset/problem/1375/D）mex|construction|sorting
1348D（https://codeforces.com/problemset/problem/1348/D）bin|construction
1554D（https://codeforces.com/problemset/problem/1554/D）construction|floor
1788C（https://codeforces.com/problemset/problem/1788/C）construction
1367D（https://codeforces.com/problemset/problem/1367/D）reverse_thinking|implemention|construction
1485D（https://codeforces.com/problemset/problem/1485/D）data_range|construction
1722G（https://codeforces.com/problemset/problem/1722/G）odd_even|xor_property|construction
1822D（https://codeforces.com/contest/1822/problem/D）construction|prefix_sum|mod|permutation
1509D（https://codeforces.com/contest/1509/problem/D）lcs|shortest_common_hypersequence|construction|data_range|O(n)|pigeonhole_principle
1473C（https://codeforces.com/contest/1473/problem/C）brain_teaser|s1s2..sn..s2s1
1469D（https://codeforces.com/contest/1469/problem/D）square|ceil|greedy|implemention
1478B（https://codeforces.com/contest/1478/problem/B）brute_force|bag_dp|construction
1682B（https://codeforces.com/contest/1682/problem/B）bitwise_and|construction|permutation_circle
1823D（https://codeforces.com/contest/1823/problem/D）greedy|construction|palindrome
1352G（https://codeforces.com/contest/1352/problem/G）construction|odd_even
1352F（https://codeforces.com/contest/1352/problem/G）construction

====================================AtCoder=====================================
AGC007B（https://atcoder.jp/contests/agc007/tasks/agc007_b）brain_teaser|math|construction
ARC086B（https://atcoder.jp/contests/abc081/tasks/arc086_b）greedy|construction|classification_discussion
ARC093B（https://atcoder.jp/contests/abc092/tasks/arc093_b）brain_teaser|construction
ABC126F（https://atcoder.jp/contests/abc126/tasks/abc126_f）brain_teaser|construction|xor_property
ABC109D（https://atcoder.jp/contests/abc109/tasks/abc109_d）odd_even|construction

"""
import math
from collections import deque, Counter, defaultdict
from typing import List

from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1478b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1478/problem/B
        tag: brute_force|bag_dp|construction|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            q, d = ac.read_list_ints()
            queries = ac.read_list_ints()
            ceil = 10 * d + 9
            dp = [0] * (ceil + 1)
            dp[0] = 1
            for i in range(1, ceil + 1):
                if str(d) in str(i):
                    for j in range(i, ceil + 1):
                        if dp[j - i]:
                            dp[j] = 1
            for num in queries:
                if num >= 10 * d + 9 or dp[num]:
                    ac.st("YES")
                else:
                    ac.st("NO")
        return

    @staticmethod
    def cf_1367d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1367/D
        tag: reverse_thinking|implemention|construction
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            m = ac.read_int()
            nums = ac.read_list_ints()
            ans = [""] * m
            lst = deque(sorted(list(s), reverse=True))
            while max(nums) >= 0:
                zero = [i for i in range(m) if nums[i] == 0]
                k = len(zero)
                while len(set(list(lst)[:k])) != 1:
                    lst.popleft()
                for i in zero:
                    nums[i] = -1
                    ans[i] = lst.popleft()
                while lst and lst[0] == ans[zero[0]]:
                    lst.popleft()
                for i in range(m):
                    if nums[i] != -1:
                        nums[i] -= sum(abs(i - j) for j in zero)
            ac.st("".join(ans))
        return

    @staticmethod
    def cf_1788c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1788/C
        tag: construction
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n % 2:
                ac.st("YES")
                x = n // 2
                for i in range(1, n + 1):
                    if i <= x:
                        ac.lst([i, i + n + x + 1])
                    else:
                        ac.lst([i, i - x + n])
            else:
                ac.st("NO")
        return

    @staticmethod
    def lc_280(nums: List[int]) -> None:
        """
        url: https://leetcode.cn/problems/wiggle-sort/
        tag: construction|sort|odd_even|classical
        """
        nums.sort()
        n = len(nums)
        ans = [0] * n
        j = n - 1
        for i in range(1, n, 2):
            ans[i] = nums[j]
            j -= 1
        j = 0
        for i in range(0, n, 2):
            ans[i] = nums[j]
            j += 1
        for i in range(n):
            nums[i] = ans[i]
        return

    @staticmethod
    def lc_1982(n: int, sums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-array-given-subset-sums/
        tag: construction|brain_teaser|classical
        """
        low = min(sums)
        if low < 0:
            sums = [num - low for num in sums]

        cnt = Counter(sums)
        lst = sorted(cnt.keys())
        cnt[0] -= 1
        ans = []
        pre = defaultdict(int)
        pre_sum = []
        for _ in range(n):
            for num in lst:
                if cnt[num] > pre[num]:
                    ans.append(num)
                    for p in pre_sum[:]:
                        pre[p + num] += 1
                        pre_sum.append(p + num)
                    pre[num] += 1
                    pre_sum.append(num)
                    break

        for i in range(1 << n):
            cur = [j for j in range(n) if i & (1 << j)]
            if sum(ans[j] for j in cur) == -low:
                for j in cur:
                    ans[j] *= -1
                return ans
        return []

    @staticmethod
    def lc_2663(s: str, k: int) -> str:
        """
        url: https://leetcode.cn/problems/lexicographically-smallest-beautiful-string/
        tag: greedy|construction|palindrome_substring|lexicographical_order|reverse_order|brute_force
        """
        n = len(s)
        for i in range(n - 1, -1, -1):
            for x in range(ord(s[i]) - ord("a") + 1, k):
                w = chr(ord("a") + x)
                if (i == 0 or s[i - 1] != w) and not (i >= 2 and w == s[i - 2]):
                    ans = s[:i] + w
                    while len(ans) < n:
                        for y in range(0, k):
                            x = chr(y + ord("a"))
                            if x != ans[-1] and (len(ans) < 2 or ans[-2] != x):
                                ans += x
                                break
                    return ans
        return ""

    @staticmethod
    def lg_p7947(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7947
        tag: greedy|construction|product_n_sum_k|prime_factorization|brain_teaser
        """
        n, k = ac.read_list_ints()
        ans = []
        for p, c in NumberTheory().get_prime_factor(n):
            ans.extend([p] * c)
        if sum(ans) > k:
            ac.st(-1)
        else:
            ans.extend([1] * (k - sum(ans)))
            ac.st(len(ans))
            ac.lst(ans)
        return

    @staticmethod
    def lg_p9101(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9101
        tag: construction|directed_graph|no_circle|classical|number_of_path
        """
        k = ac.read_int()
        ac.st(98)
        ac.lst([33, -1])
        for i in range(2, 34):
            if k & (1 << (i - 2)):
                cur = [i + 32]
            else:
                cur = [-1]
            if i > 2:
                cur.append(i - 1)
            else:
                cur.append(-1)
            ac.lst(cur)
        for i in range(34, 99):
            if i in [34, 66]:
                ac.lst([98, -1])
            elif i == 98:
                ac.lst([-1, -1])
            else:
                ac.lst([i - 1, i - 33 if i >= 67 else i + 31])
        return

    @staticmethod
    def lg_p8976(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8976
        tag: brute_force|construction|classical
        """
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            mid = n // 2 + 1
            if a + b > n * (n + 1) // 2 or ac.max(a, b) > (n // 2) * (mid + n) // 2:
                ac.st(-1)
                continue
            s = n * (n + 1) // 2
            lst = [a, b]
            ans = []
            for i in range(n // 2 + 1):
                if ans:
                    break
                x = n // 2 - i
                for aa, bb in [[0, 1], [1, 0]]:
                    if x:
                        rest = lst[aa] - i * (i + 1) // 2
                        y = math.ceil((rest * 2 / x - x + 1) / 2)
                        y = ac.max(y, i + 1)

                        if y + x - 1 <= n:
                            cur = i * (i + 1) // 2 + x * (y + y + x - 1) // 2
                            if cur >= lst[aa] and s - cur >= lst[bb]:
                                pre = list(range(1, i + 1)) + list(range(y, y + x))
                                post = list(range(i + 1, y)) + list(range(y + x, n + 1))
                                ans = pre + post if aa == 0 else post + pre
                                break
                    else:
                        if n // 2 * (1 + n // 2) // 2 >= a and s - a >= b:
                            ans = list(range(1, n + 1))
                            break
            if not ans:
                ac.st(-1)
            else:
                ac.lst(ans)
        return

    @staticmethod
    def lg_p8910(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8910
        tag: permutation_circle|construction|classical|brain_teaser
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nex = [0] * (n + 1)
            for i in range(k):
                nex[i] = n - k + i
            for i in range(k, n):
                nex[i] = i - k
            ans = []
            for i in range(n):
                if nex[i] != i:
                    lst = [i]
                    while nex[lst[-1]] != i:
                        lst.append(nex[lst[-1]])
                    m = len(lst)
                    ans.append([n + 1, lst[0] + 1])
                    for x in range(1, m):
                        ans.append([lst[x - 1] + 1, lst[x] + 1])
                    ans.append([lst[m - 1] + 1, n + 1])
                    for x in lst:
                        nex[x] = x
            ac.st(len(ans))
            for a in ans:
                ac.lst(a)
        return

    @staticmethod
    def lg_p8880(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8880
        tag: brain_teaser|construction|odd_even|classical
        """
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(-1)
            return
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(nums)}
        a = [-1] * n
        b = [-1] * n
        for i in range(n):
            j = (i - 1) % n
            x = (i + j) % n
            a[ind[x]] = i
            b[ind[x]] = j
        ac.lst(a)
        ac.lst(b)
        return

    @staticmethod
    def cf_1823d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1823/problem/D
        tag: greedy|construction|palindrome
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            x = [0] + [x - 3 for x in ac.read_list_ints()]
            c = [0] + [x - 3 for x in ac.read_list_ints()]
            st = "abc"
            ans = ["abc"]
            ind = 0
            for i in range(k):
                dx = x[i + 1] - x[i]
                dc = c[i + 1] - c[i]
                if dx < dc:
                    ac.st("NO")
                    break
                ans.append(chr(ord("d") + i) * dc)
                for _ in range(dx - dc):
                    ans.append(st[ind])
                    ind += 1
                    ind %= 3
            else:
                ac.st("YES")
                ac.st("".join(ans))
        return

    @staticmethod
    def cf_1722g(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1722/G
        tag: odd_even|xor_property|construction
        """

        # def sum_xor(n):
        #     """xor num of range(0, x+1)"""
        #     if n % 4 == 0:
        #         return n  # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        #     elif n % 4 == 1:
        #         return 1  # n^(n-1)
        #     elif n % 4 == 2:
        #         return n + 1  # n^(n-1)^(n-2)
        #     return 0  # n^(n-1)^(n-2)^(n-3)

        for _ in range(ac.read_int()):
            n = ac.read_int()  # n >= 3
            if n % 4 == 0:
                ans = list(range(n))
            elif n % 4 == 1:
                ans = [0] + list(range(2, n + 1))
            elif n % 4 == 2:
                ans = list(range(1, n - 1)) + [1 << 20, (1 << 20) | (n - 2)]
            else:
                ans = list(range(1, n + 1))
            ac.lst(ans)
        return
