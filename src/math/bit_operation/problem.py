"""

Algorithm：bit_operation
Description：bit_wise|xor|or|and|brute_force
Property：(4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0  (2*n)^(2*n+1)=1 (a&b)^(a&c) = a&(b^c)

====================================LeetCode====================================
201（https://leetcode.cn/problems/bitwise-and-of-numbers-range/）brain_teaser
2354（https://leetcode.cn/problems/number-of-excellent-pairs/）brain_teaser|hash|counter|brute_force
260（https://leetcode.cn/problems/single-number-iii/）bit_operation|cor_property|lowest_bit
2571（https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/）operation|bit_property
2568（https://leetcode.cn/problems/minimum-impossible-or/）greedy
2564（https://leetcode.cn/problems/substring-xor-queries/）bit_operation|bit_property
1238（https://leetcode.cn/problems/circular-permutation-in-binary-representation/）gray_code|classical
89（https://leetcode.cn/problems/gray-code/）gray_code|classical
137（https://leetcode.cn/problems/single-number-ii/）bit_operation|counter
56（https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/）bit_operation|counter
260（https://leetcode.cn/problems/single-number-iii/）bit_operation|counter
2546（https://leetcode.cn/problems/apply-bitwise-operations-to-make-strings-equal/）xor_property|brain_teaser
1486（https://leetcode.cn/problems/xor-operation-in-an-array/）xor_property
1734（https://leetcode.cn/problems/decode-xored-permutation/）math|xor_property|odd_xor
1787（https://leetcode.cn/problems/make-the-xor-of-all-segments-equal-to-zero/）xor_property|data_range|brute_force
1835（https://leetcode.cn/problems/find-xor-sum-of-all-pairs-bitwise-and/）bit_operation|implemention
1611（https://leetcode.cn/problems/minimum-one-bit-operations-to-make-integers-zero/）gray_code|classical
2275（https://leetcode.cn/problems/largest-combination-with-bitwise-and-greater-than-zero/）range_add|classical|st
2527（https://leetcode.cn/problems/find-xor-beauty-of-array/description/）brute_force|brain_teaser|bit_operation
2680（https://leetcode.cn/problems/maximum-or/description/）greedy|brute_force|prefix_suffix
100087（https://leetcode.cn/problems/apply-operations-on-array-to-maximize-sum-of-squares/）bit_wise|bit_operation|greedy
3007（https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/）bit_operation|binary_search|bit_operation|binary_search|digital_dp
100179（https://leetcode.cn/problems/minimize-or-of-remaining-elements-using-operations/）bit_operation|greedy|brain_teaser
3145（https://leetcode.cn/problems/find-products-of-elements-of-big-array/description/）bit_operation|data_range|classical|inclusion_exclusion|counter
233（https://leetcode.cn/problems/number-of-digit-one/description/）bit_operation|digital_dp|circular_section
3315（https://leetcode.cn/problems/construct-the-minimum-bitwise-array-ii/）construction|guess_table|bit_operation
3141（https://leetcode.cn/problems/maximum-hamming-distances/）bfs|brain_teaser|build_graph|bit_operation
3215（https://leetcode.cn/problems/count-triplets-with-even-xor-set-bits-ii/）bit_operation|odd_even|xor_property

=====================================LuoGu======================================
P5657（https://www.luogu.com.cn/problem/P5657）bit_operation
P6102（https://www.luogu.com.cn/problem/P6102）bit_operation|and
P7442（https://www.luogu.com.cn/problem/P7442）bit_operation|implemention|observe_pattern
P7617（https://www.luogu.com.cn/problem/P7617）bit_operation|brute_force
P7627（https://www.luogu.com.cn/problem/P7627）bit_operation|brute_force
P7649（https://www.luogu.com.cn/problem/P7649）3-base|greedy|implemention
P1582（https://www.luogu.com.cn/problem/P1582）base|brain_teaser
P2114（https://www.luogu.com.cn/problem/P2114）bit_operation|implemention|greedy
P2326（https://www.luogu.com.cn/problem/P2326）bit_operation|implemention|greedy|maximum_and
P4144（https://www.luogu.com.cn/problem/P4144）bit_operation|greedy|brain_teaser
P4310（https://www.luogu.com.cn/problem/P4310）linear_dp|bit_operation
P5390（https://www.luogu.com.cn/problem/P5390）bit_operation
P6824（https://www.luogu.com.cn/problem/P6824）bit_operation|xor|diff_array|action_scope|counter
P8842（https://www.luogu.com.cn/problem/P8842）prime_factorization|prefix_sum|counter
P8965（https://www.luogu.com.cn/problem/P8965）tree_dp|xor
P3760（https://www.luogu.com.cn/problem/P3760）bit_operation|contribution_method|classical|two_pointers|sorted_list|brain_teaser
U360500（https://www.luogu.com.cn/problem/U360500）bit_operation|contribution_method|classical|two_pointers|sorted_list|brain_teaser
P5390（ttps://www.luogu.com.cn/problem/P5390）bit_operation|contribution_method

===================================CodeForces===================================
305C（https://codeforces.com/problemset/problem/305/C）2-base
878A（https://codeforces.com/problemset/problem/878/A）bit_operation
282C（https://codeforces.com/problemset/problem/282/C）bit_operation
1554C（https://codeforces.com/contest/1554/problem/C）bit_operation|greedy
1800F（https://codeforces.com/contest/1800/problem/F）bit_operation|brute_force|counter
276D（https://codeforces.com/contest/276/problem/D）maximum_xor|classical
1742G（https://codeforces.com/contest/1742/problem/G）prefix_or|lexicographical_order|construction|specific_plan
1851F（https://codeforces.com/contest/1851/problem/F）minimum_xor_pair|classical|sort|adjacent_pair|brain_teaser
1879D（https://codeforces.com/contest/1879/problem/D）bit_operation|bit_contribution_method|prefix_sum|counter|prefix_or
1368D（https://codeforces.com/problemset/problem/1368/D）implemention|greedy|bit_wise|bit_operation
1802C（https://codeforces.com/contest/1802/problem/C）construction|xor_property
1918C（https://codeforces.com/contest/1918/problem/C）greedy|bit_operation
1669H（https://codeforces.com/contest/1669/problem/H）bit_operation
1760G（https://codeforces.com/contest/1760/problem/G）bit_operation|dfs|brute_force
1066E（https://codeforces.com/contest/1066/problem/E）bit_operation|brute_force|implemention|prefix_sum
1790E（https://codeforces.com/contest/1790/problem/E）bit_operation
1968F（https://codeforces.com/contest/1968/problem/F）brute_force|bit_operation|binary_search
1973B（https://codeforces.com/contest/1973/problem/B）bit_operation|implemention|greedy
1362C（https://codeforces.com/problemset/problem/1362/C）bit_count|bit_operation
1981B（https://codeforces.com/contest/1981/problem/B）bit_operation|classical|range_or
1285D（https://codeforces.com/problemset/problem/1285/D）bitwise_xor|minimax|divide_and_conquer
1982E（https://codeforces.com/contest/1982/problem/E）divide_and_conquer|bit_operation|brain_teaser|segment_tree
1303D（https://codeforces.com/problemset/problem/1303/D）bit_operation|greedy|implemention
1466E（https://codeforces.com/problemset/problem/1466/E）bit_operation|math|contribution_method|classical
1491D（https://codeforces.com/problemset/problem/1491/D）bit_operation|observation|brain_teaser
1322B（https://codeforces.com/problemset/problem/1322/B）bit_operation|contribution_method|classical|two_pointers
1557C（https://codeforces.com/problemset/problem/1557/C）bit_operation|brain_teaser|brute_force|observation
1592C（https://codeforces.com/contest/1592/problem/C）bit_operation|tree_xor|construction|observation
1720D1（https://codeforces.com/problemset/problem/1720/D1）linear_dp|data_range|observation|bit_operation
1632C（https://codeforces.com/problemset/problem/1632/C）bit_operation|brute_force|construction
2020C（https://codeforces.com/contest/2020/problem/C）bit_operation|construction
1715D（https://codeforces.com/problemset/problem/1715/D）greedy|construction|bit_operation
1416C（https://codeforces.com/problemset/problem/1416/C）bit_operation|divide_and_conquer|reverse_pair
2036F（https://codeforces.com/contest/2036/problem/F）bit_property|bit_operation|inclusion_exclusion|classical
2035C（https://codeforces.com/contest/2035/problem/C）construction|bit_operation
2020C（https://codeforces.com/contest/2020/problem/C）bit_operation|construction|observation

====================================AtCoder=====================================
ABC117D（https://atcoder.jp/contests/abc117/tasks/abc117_d）bit_operation|greedy|brain_teaser
ABC147D（https://atcoder.jp/contests/abc147/tasks/abc147_d）classical|xor_sum
ABC121D（https://atcoder.jp/contests/abc121/tasks/abc121_d）classical|xor_sum
ABC308G（https://atcoder.jp/contests/abc308/tasks/abc308_g）minimum_pair_xor|dynamic
ABC281F（https://atcoder.jp/contests/abc281/tasks/abc281_f）bit_operation|sort|binary_trie|greedy|dfs|implemention|divide_conquer|merge
ABC261E（https://atcoder.jp/contests/abc261/tasks/abc261_e）bit_operation|brain_teaser|implemention|classical
ABC356D（https://atcoder.jp/contests/abc356/tasks/abc356_d）bit_count|classical|math|digital_dp
ARC092B（https://atcoder.jp/contests/arc092/tasks/arc092_b）bit_operation|contribution_method|classical|two_pointers
ABC365E（https://atcoder.jp/contests/abc365/tasks/abc365_e）bit_operation|prefix_sum
ABC201E（https://atcoder.jp/contests/abc201/tasks/abc201_e）bit_operation|classical

=====================================AcWing=====================================
998（https://www.acwing.com/problem/content/1000/）or|xor|and|bit_operation|greedy
4614（https://www.acwing.com/problem/content/4617/）bit_operation|brute_force|prefix_sum|preprocess

=====================================Library=====================================
1（https://ac.nowcoder.com/acm/contest/53485/F）minimum_pair_xor|dynamic|classical
2（https://www.codechef.com/problems/LEXMAX）bit_operation|maximum_and|lexicographically_maximal|prefix_and|greedy|classical
3（https://www.codechef.com/problems/PREFSUFF）bit_operation|construction

https://blog.csdn.net/qq_35473473/article/details/106320878

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/MEXXOR?tab=statement）bit_operation|brain_teaser

"""
import math
from collections import defaultdict, Counter
from functools import lru_cache
from functools import reduce
from operator import xor, or_
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.math.bit_operation.template import BitOperation, MinimumPairXor
from src.math.comb_perm.template import Combinatorics
from src.util.fast_io import FastIO


class Solution:

    def __int__(self):
        return

    @staticmethod
    def cf_1742g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1742/problem/G
        tag: prefix_or|lexicographical_order|construction|specific_plan
        """

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            tot = reduce(or_, nums)
            pre = 0
            ans = []
            for i in range(n):
                if pre == tot or len(ans) == n:
                    break
                nex = val = -1
                for j in range(n):
                    if nums[j] >= 0 and pre | nums[j] > val:
                        val = pre | nums[j]
                        nex = j
                pre |= nums[nex]
                ans.append(nums[nex])
                nums[nex] = -1
            for i in range(n):
                if nums[i] >= 0:
                    ans.append(nums[i])
            ac.lst(ans)
        return

    @staticmethod
    def cf_276d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/276/problem/D
        tag: maximum_xor|classical|hard|brain_teaser
        """

        a, b = ac.read_list_ints()
        n = len(bin(b)) - 2
        ans = 0
        for i in range(n - 1, -1, -1):
            if b - a >= (1 << i):
                ans |= (1 << i)
            else:
                if (a ^ b) & (1 << i):
                    ans |= (1 << i)
        ac.st(ans)
        return

    @staticmethod
    def cf_1800f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1800/problem/F
        tag: bit_operation|brute_force|counter|odd_even|brain_teaser|hard
        """

        ac.get_random_seed()
        n = ac.read_int()
        ans = 0
        mask = (1 << 26) - 1
        state = [mask ^ (1 << j) for j in range(26)]
        dct = dict()
        for _ in range(n):
            a = b = 0
            for w in ac.read_str():
                i = ord(w) - ord("a")
                a ^= (1 << i)
                b |= (1 << i)
            if a ^ ac.random_seed not in dct:
                dct[a ^ ac.random_seed] = [0] * 26
            for j in range(26):
                if not b & (1 << j):
                    if a ^ state[j] ^ ac.random_seed in dct:
                        ans += dct[a ^ state[j] ^ ac.random_seed][j]
                    dct[a ^ ac.random_seed][j] += 1
        ac.st(ans)
        return

    @staticmethod
    def lc_260(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/single-number-iii/
        tag: bit_operation|counter|brain_teaser
        """

        s = reduce(xor, nums)
        last = s & (-s)
        one = two = 0
        for num in nums:
            if num & last:
                one ^= num
            else:
                two ^= num
        return [one, two]

    @staticmethod
    def lc_137(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/single-number-ii/
        tag: bit_operation|counter|classical
        """

        floor = (1 << 31) + 1
        dp = [0] * 33
        for num in nums:
            num += floor
            for i in range(33):
                if num & (1 << i):
                    dp[i] += 1
        ans = 0
        for i in range(33):
            if dp[i] % 3:
                ans |= (1 << i)
        return ans - floor

    @staticmethod
    def cf_1554c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1554/problem/C
        tag: bit_operation|greedy|mex_like|brain_teaser|classical|hard|reverse_thinking
        """

        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            assert 0 <= n <= 10 ** 9
            assert 0 <= m <= 10 ** 9
            p = m + 1
            ans = 0  # n^ans >= m+1
            for i in range(30, -1, -1):
                if ans ^ n >= p:
                    break
                if n & (1 << i) == p & (1 << i):
                    continue
                if p & (1 << i):
                    ans |= (1 << i)
            ac.st(ans)
        return

    @staticmethod
    def lc_1787(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/make-the-xor-of-all-segments-equal-to-zero/
        tag: xor_property|data_range|brute_force|hard
        """
        m = max(len(bin(num)) - 2 for num in nums)
        pre = [math.inf] * (1 << m)
        pre[0] = 0
        for i in range(k):
            lst = nums[i::k]
            n = len(lst)
            cnt = Counter(lst)
            low = min(pre)
            cur = [low + n for _ in pre]
            for j in range(1 << m):
                for num in cnt:
                    a, b = cur[j], pre[j ^ num] + n - cnt[num]
                    cur[j] = a if a < b else b
            pre = cur[:]
        return pre[0]

    @staticmethod
    def lc_2568(nums):
        """
        url: https://leetcode.cn/problems/minimum-impossible-or/
        tag: greedy|brain_teaser
        """

        dct = set(nums)
        ans = 1
        while ans in dct:
            ans *= 2
        return ans

    @staticmethod
    def lc_2571_1(n):
        """
        url: https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/
        tag: operation|bit_property
        """

        @lru_cache(None)
        def dfs(k):
            if not k:
                return 0
            if k.bit_count() == 1:
                return 1
            low = k & (-k)
            return 1 + min(dfs(k - low), dfs(k + low))

        return dfs(n)  # (3 * n ^ n).bit_count()

    @staticmethod
    def lc_2571_2(n):
        """
        url: https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/
        tag: operation|bit_property
        """
        ans = cnt = 0
        m = n.bit_length()
        for i in range(m):
            if n & (1 << i):
                cnt += 1
            else:
                if cnt == 1:
                    ans += 1
                    cnt = 0
                elif cnt >= 2:
                    if n & (1 << (i + 1)):
                        ans += 1
                        cnt = 1
                    else:
                        ans += 2
                        cnt = 0
        if cnt:
            ans += 1 if cnt == 1 else 2
        return ans

    @staticmethod
    def lc_2275(candidates: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/largest-combination-with-bitwise-and-greater-than-zero/
        tag: range_add|classical|st
        """

        count = [0] * 24
        for num in candidates:
            for i in range(24):
                if num & (1 << i):
                    count[i] += 1
        return max(count)

    @staticmethod
    def lc_2564(s, queries):
        """
        url: https://leetcode.cn/problems/substring-xor-queries/
        tag: bit_operation|bit_property|data_range
        """

        dct = defaultdict(set)
        m = len(queries)
        for i in range(m):
            a, b = queries[i]
            x = bin(a ^ b)[2:]
            dct[x].add(i)
        ceil = max(len(x) for x in dct)
        ans = [[-1, -1] for _ in range(m)]
        n = len(s)
        for i in range(n):
            for j in range(max(i - ceil + 1, 0), i + 1):
                st = s[j:i + 1]
                if dct[st]:
                    for k in dct[st]:
                        ans[k] = [j, i]
                    dct[st] = set()
        return ans

    @staticmethod
    def lc_1238(n: int, start: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/circular-permutation-in-binary-representation/
        tag: gray_code|classical
        """

        ans = BitOperation().get_graycode(n)
        i = ans.index(start)
        return ans[i:] + ans[:i]

    @staticmethod
    def lc_89(n: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/gray-code/
        tag: gray_code|classical
        """

        ans = BitOperation().get_graycode(n)
        return ans

    @staticmethod
    def abc_117d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc117/tasks/abc117_d
        tag: bit_operation|greedy|brain_teaser|implemention|hard|bit_property
        """

        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = pre = 0
        for i in range(40, -1, -1):
            cnt = Counter(int(num & (1 << i) > 0) for num in nums)
            if cnt[1] >= cnt[0] or pre + (1 << i) > k:  # half|bit
                ans += cnt[1] * (1 << i)
            else:
                pre += (1 << i)
                ans += cnt[0] * (1 << i)
        ac.st(ans)
        return

    @staticmethod
    def abc_121d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc121/tasks/abc121_d
        tag: classical|xor_sum
        """
        a, b = ac.read_list_ints()
        ans = BitOperation().sum_xor(b)
        if a:
            ans ^= BitOperation().sum_xor((a - 1))
        ac.st(ans)
        return

    @staticmethod
    def ac_998(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/1000/
        tag: or|xor|and|bit_operation|greedy|implemention
        """

        n, m = ac.read_list_ints()
        ans = [[0, 1 << i] for i in range(32)]
        for _ in range(n):
            op, t = ac.read_list_strs()
            t = int(t)
            if op == "AND":
                for i in range(32):
                    ans[i][0] &= t & (1 << i)
                    ans[i][1] &= t & (1 << i)
            elif op == "OR":
                for i in range(32):
                    ans[i][0] |= t & (1 << i)
                    ans[i][1] |= t & (1 << i)
            else:
                for i in range(32):
                    ans[i][0] ^= t & (1 << i)
                    ans[i][1] ^= t & (1 << i)
        res = x = 0
        for i in range(31, -1, -1):
            if ans[i][1] > ans[i][0] and (x | (1 << i)) <= m:
                x |= (1 << i)
                res += ans[i][1]
            else:
                res += ans[i][0]
        ac.st(res)
        return

    @staticmethod
    def lg_p1582(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1582
        tag: base|brain_teaser
        """
        n, k = ac.read_list_ints()
        ans = 0
        while n.bit_count() > k:
            ans += n & (-n)
            n += n & (-n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2114(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2114
        tag: bit_operation|implemention|greedy
        """

        n, m = ac.read_list_ints()
        one = [1 << i for i in range(32)]
        zero = [0] * 32
        for _ in range(n):
            op, t = ac.read_list_strs()
            t = int(t)
            for i in range(32):
                if op == "AND":
                    one[i] &= (t & (1 << i))
                    zero[i] &= (t & (1 << i))
                elif op == "OR":
                    one[i] |= (t & (1 << i))
                    zero[i] |= (t & (1 << i))
                else:
                    one[i] ^= (t & (1 << i))
                    zero[i] ^= (t & (1 << i))

        ans = 0
        for i in range(31, -1, -1):
            if one[i] > zero[i] and m >= (1 << i):
                m -= (1 << i)
                ans += one[i]
            else:
                ans += zero[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2326(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2326
        tag: bit_operation|implemention|greedy|maximum_and
        """

        for case in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ans = 0
            while nums:
                cnt = [0] * 32
                for num in nums:
                    for i in range(21):
                        if num & (1 << i):
                            cnt[i] += 1
                for i in range(20, -1, -1):
                    if cnt[i] >= 2:
                        ans |= (1 << i)
                        nums = [num ^ (1 << i) for num in nums if num & (1 << i) and num ^ (1 << i)]
                        break
                else:
                    nums = []
            ac.st(f"Case #{case + 1}: {ans}")
        return

    @staticmethod
    def lg_p4144(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4144
        tag: bit_operation|greedy|brain_teaser|hard|classical
        """

        n, b, p = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = max(nums) * 2
        ac.st(pow(ans + 233, b, p))
        return

    @staticmethod
    def lg_p4310(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4310
        tag: linear_dp|bit_operation|classical
        """

        ac.read_int()
        nums = ac.read_list_ints()
        cnt = [0] * 32
        for num in nums:
            pre = 0
            lst = []
            for j in range(32):
                if num & (1 << j):
                    lst.append(j)
                    pre = max(pre, cnt[j])
            pre += 1
            for j in lst:
                cnt[j] = pre
        ac.st(max(cnt))
        return

    @staticmethod
    def lg_p5390(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5390
        url: https://www.luogu.com.cn/problem/U360642
        tag: bit_operation|odd_even|classical
        """

        mod = 998244353
        for _ in range(ac.read_int()):
            nums = ac.read_list_ints()
            n = nums[0]
            num = reduce(or_, nums[1:])
            pp = pow(2, n - 1, mod)
            ans = 0
            for i in range(32):
                if num & (1 << i):
                    ans += (1 << i) * pp
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p6824(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6824
        tag: bit_operation|xor|diff_array|action_scope|counter|classical|hard
        """

        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        m = len(bin(max(k, max(nums)))) - 1
        diff = [0] * ((1 << m) + 1)
        for a in nums:
            pre = 0
            for i in range(m - 1, -1, -1):
                if k & (1 << i):
                    if a & (1 << i):
                        low = pre ^ (1 << i)
                        high = low ^ ((1 << i) - 1)
                        diff[low] += 1
                        diff[high + 1] -= 1
                    else:
                        low, high = pre, pre ^ ((1 << i) - 1)
                        diff[low] += 1
                        diff[high + 1] -= 1
                        pre ^= (1 << i)
                else:
                    if a & (1 << i):
                        pre ^= (1 << i)
            diff[pre] += 1
            diff[pre + 1] -= 1
        ac.st(max(ac.accumulate(diff)))
        return

    @staticmethod
    def lg_p8842(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8842
        tag: prime_factorization|prefix_sum|counter|classical|hard
        """

        n = 1 << 21
        prime = [0] * (n + 1)
        prime[0] = 0
        prime[1] = 1
        for i in range(2, n + 1):
            if not prime[i]:
                for j in range(i * i, n + 1, i):
                    prime[j] = 1
            prime[i] += prime[i - 1]
        for _ in range(ac.read_int()):
            x = ac.read_int()
            ans = 0
            for k in range(21):
                if x & (1 << k):
                    ans += (1 << (k + 1)) - (1 << k) - (prime[(1 << (k + 1)) - 1] - prime[(1 << k) - 1])
            ac.st(ans)
        return

    @staticmethod
    def lc_1486(n: int, start: int) -> int:
        """
        url: https://leetcode.cn/problems/xor-operation-in-an-array/
        tag: xor_property|hard
        """

        s = start // 2
        bo = BitOperation()
        e = n & start & 1
        return (bo.sum_xor(s - 1) ^ bo.sum_xor(s + n - 1)) * 2 + e

    @staticmethod
    def lc_1734(encoded: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/decode-xored-permutation/
        tag: math|xor_property|odd_xor
        """

        n = len(encoded) + 1
        total = BitOperation().sum_xor(n)
        odd = reduce(xor, encoded[1::2])
        ans = [total ^ odd]
        for num in encoded:
            ans.append(ans[-1] ^ num)
        return ans

    @staticmethod
    def ac_4614(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4617/
        tag: bit_operation|brute_force|prefix_sum|preprocess|brain_teaser|classical
        """

        n, m, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = [ac.read_str() for _ in range(m)]
        cnt = Counter(lst)
        dct = [0] * (1 << n)
        for va in cnt:
            dct[int("0b" + va, 2)] = cnt[va]

        weight = [0] * (1 << n)
        for i in range(1 << n):
            weight[i] = sum(nums[j] for j in range(n) if not i & (1 << (n - 1 - j)))

        res = [[0] * 101 for _ in range(1 << n)]
        for i in range(1 << n):
            for j in range(1 << n):
                s = weight[i ^ j]
                if s <= 100:
                    res[i][s] += dct[j]

        for i in range(1 << n):
            for j in range(1, 101):
                res[i][j] += res[i][j - 1]

        for _ in range(q):
            t, k = ac.read_list_strs()
            k = int(k)

            ac.st(res[int("0b" + t, 2)][k])
        return

    @staticmethod
    def lc_3007(k: int, x: int) -> int:

        """
        url: https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/description/
        tag: bit_operation|binary_search|digital_dp
        """

        def check(num):
            num += 1
            cnt = 0
            for i in range(x, 64, x):
                n = 1 << (i - 1)
                if n > num:
                    break
                cnt += (num // (n << 1)) * n
                if num % (n << 1) > n:
                    cnt += (num % (n << 1)) - n
            return cnt <= k

        ans = BinarySearch().find_int_right(0, 10 ** 15, check)
        return ans

    @staticmethod
    def abc_308g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc308/tasks/abc308_g
        tag: minimum_pair_xor|dynamic|classical
        """
        q = ac.read_int()
        minimum_xor = MinimumPairXor()
        for _ in range(q):
            lst = ac.read_list_ints()

            if lst[0] == 1:
                minimum_xor.add(lst[1])
            elif lst[0] == 2:
                minimum_xor.remove(lst[1])
            else:
                ac.st(minimum_xor.query())
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://ac.nowcoder.com/acm/contest/53485/F
        tag: minimum_pair_xor|dynamic|classical
        """
        q = ac.read_int()
        minimum_xor = MinimumPairXor()
        for _ in range(q):
            lst = ac.read_list_strs()

            if lst[0] == "ADD":
                minimum_xor.add(int(lst[1]))
            elif lst[0] == "DEL":
                minimum_xor.remove(int(lst[1]))
            else:
                ac.st(minimum_xor.query())
        return

    @staticmethod
    def lc_100179(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/minimize-or-of-remaining-elements-using-operations/
        tag: bit_operation|greedy|brain_teaser
        """
        ans = mask = 0
        for i in range(max(nums).bit_length(), -1, -1):
            mask |= 1 << i
            pre = -1
            cnt = 0
            for num in nums:
                pre &= num & mask
                if pre == 0:
                    pre = -1
                else:
                    cnt += 1
            if cnt > k:
                ans |= 1 << i
                mask ^= 1 << i
        return ans

    @staticmethod
    def cf_1918c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1918/problem/C
        tag: bit_operation|greedy
        """
        for _ in range(ac.read_int()):
            a, b, r = ac.read_list_ints()
            for i in range((a ^ b).bit_length() - 2, -1, -1):
                bit = 1 << i
                if (a ^ b) & bit and bit <= r and abs((a ^ bit) - (b ^ bit)) < abs(a - b):
                    a ^= bit
                    b ^= bit
                    r -= bit
            ac.st(abs(a - b))
        return

    @staticmethod
    def abc_281f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc281/tasks/abc281_f
        tag: bit_operation|sort|binary_trie|greedy|dfs|implemention|divide_conquer|merge
        """
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        n = len(nums)

        def dfs(i, ll, rr):
            if i == -1:
                return 0
            mid = -1
            for j in range(ll, rr + 1):
                if nums[j] & (1 << i):
                    mid = j
                    break
            if mid == -1 or mid == ll:
                return dfs(i - 1, ll, rr)

            return min(dfs(i - 1, ll, mid - 1), dfs(i - 1, mid, rr)) | (1 << i)

        ans = dfs(29, 0, n - 1)
        ac.st(ans)
        return

    @staticmethod
    def abc_261e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc261/tasks/abc261_e
        tag: bit_operation|brain_teaser|implemention|classical
        """
        n, c = ac.read_list_ints()
        o = (1 << 31) - 1
        z = 0
        for _ in range(n):
            t, a = ac.read_list_ints()
            if t == 1:
                o &= a
                z &= a
            elif t == 2:
                o |= a
                z |= a
            else:
                o ^= a
                z ^= a
            c = (c & o) | (~c & z)
            ac.st(c)
        return

    @staticmethod
    def cc_2(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/LEXMAX
        tag: bit_operation|maximum_and|lexicographically_maximal|prefix_and|greedy|classical
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ans = [max(nums)]
            nums.remove(ans[0])
            while nums:
                val = max(num & ans[-1] for num in nums)
                cnt = sum((num & ans[-1]) == val for num in nums)
                ans.extend([val] * cnt)
                nums = [num for num in nums if (num & ans[-1]) != val]
            ac.lst(ans)
        return

    @staticmethod
    def abc_356d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc356/tasks/abc356_d
        tag: bit_count|classical|math|digital_dp
        """
        n, m = ac.read_list_ints()
        mod = 998244353
        ans = 0
        for i in range(61):
            if m & (1 << i):
                circle = 1 << (i + 1)
                ans += (1 << i) * ((n + 1) // circle)
                if (n + 1) % circle > (1 << i):
                    ans += (n + 1) % circle - (1 << i)
                ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1981b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1981/problem/B
        tag: bit_operation|classical|range_or
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()

            low = max(n - m, 0)
            high = n + m
            ans = 0
            for i in range(64):
                cc = 1 << (i + 1)
                pp = 1 << i
                if 1 <= (high + 1) % cc <= pp and 1 <= (low + 1) % cc <= pp and (high + 1) // cc == (low + 1) // cc:
                    continue
                ans |= 1 << i
            ac.st(ans)
        return

    @staticmethod
    def cf_1285d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1285/D
        tag: bitwise_xor|minimax|divide_and_conquer
        """
        ac.read_int()

        nums = ac.read_list_ints()

        def check(lst, bit):
            if bit == -1:
                return 0
            one = []
            zero = []
            for num in lst:
                if (num >> bit) & 1:
                    one.append(num)
                else:
                    zero.append(num)
            if not one:
                return check(zero, bit - 1)
            if not zero:
                return check(one, bit - 1)
            return min(check(one, bit - 1), check(zero, bit - 1)) + (1 << bit)

        ans = check(nums, 29)
        ac.st(ans)
        return

    @staticmethod
    def cf_1982e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1982/problem/E
        tag: divide_and_conquer|bit_operation|brain_teaser|segment_tree
        """
        mod = 10 ** 9 + 7

        def compute(x):
            return (x * (x + 1) // 2) % mod

        @lru_cache(None)
        def dfs(nn, kk):
            m = nn.bit_length()
            if m <= kk:
                return compute(nn + 1), nn + 1, nn + 1
            if nn == 0:
                return compute(1), 1, 1
            if kk == 0:
                return compute(1), 1, 0
            m -= 1
            res1, s1, e1 = dfs((1 << m) - 1, kk)
            res2, s2, e2 = dfs(nn - (1 << m), kk - 1)
            res1 -= compute(s1)
            if not s1 == e1 == 1 << m:
                res1 -= compute(e1)

            res2 -= compute(s2)
            if not s2 == e2 == nn - (1 << m) + 1:
                res2 -= compute(e2)

            res = res1 + res2
            if s1 == e1 == 1 << m and s2 == e2 == nn - (1 << m) + 1:
                res += compute(s1 + s2)
                return res % mod, s1 + s2, e1 + e2
            elif s1 == e1 == 1 << m:
                res += compute(s1 + s2) + compute(e2)
                return res % mod, s1 + s2, e2
            elif s2 == e2 == nn - (1 << m) + 1:
                res += compute(s1) + compute(e1 + e2)
                return res % mod, s1, e2 + e1

            res += compute(s1) + compute(s2) + compute(e1) + compute(e2)
            return res % mod, s1, e2

        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            ans = dfs(n - 1, k)[0]
            ac.st(ans)
        return

    @staticmethod
    def cf_1303d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1303/D
        tag: bit_operation|greedy|implemention
        """

        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            cnt = [0] * 64
            for num in nums:
                cnt[num.bit_length() - 1] += 1
            ans = 0
            for x in range(64):
                if x:
                    cnt[x] += cnt[x - 1] // 2
                if not n & (1 << x):
                    continue
                if cnt[x] == 0:
                    for i in range(x + 1, 64):
                        if cnt[i]:
                            for j in range(i - 1, x - 1, -1):
                                ans += 1
                                cnt[j + 1] -= 1
                                cnt[j] += 2
                            break
                if cnt[x]:
                    cnt[x] -= 1
                    n ^= (1 << x)
                else:
                    break
            if n:
                ac.st(-1)
            else:
                ac.st(ans)
        return

    @staticmethod
    def cf_1466e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1466/E
        tag: bit_operation|math|contribution_method|classical
        """
        mod = 10 ** 9 + 7

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            ans = 0
            cnt = [0] * 61
            for x in range(61):
                for num in nums:
                    if (num >> x) & 1:
                        cnt[x] += 1
            for num in nums:
                left = right = 0
                for x in range(61):
                    if (num >> x) & 1:
                        left += (1 << x) * cnt[x]
                        right += n * (1 << x)
                    else:
                        right += cnt[x] * (1 << x)
                ans += left * right
                ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def cf_1491d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1491/D
        tag: bit_operation|observation|brain_teaser
        """
        for _ in range(ac.read_int()):
            u, v = ac.read_list_ints()
            if u > v:
                ac.no()
                continue
            if u == v:
                ac.yes()
                continue
            uu = vv = 0
            for i in range(30):
                if (u >> i) & 1:
                    uu += 1
                if (v >> i) & 1:
                    vv += 1
                if uu < vv:
                    ac.no()
                    break
            else:
                ac.yes()
        return

    @staticmethod
    def cf_1322b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1322/B
        tag: bit_operation|contribution_method|classical|two_pointers
        """
        max_bit = 25
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        for x in range(max_bit):
            one, zero = [], []
            for num in nums:
                if (num >> x) & 1:
                    one.append(num)
                else:
                    zero.append(num)
            nums = zero + one

            mask = (1 << (x + 1)) - 1
            tmp = [(1 << x, (1 << (x + 1)) - 1), ((1 << x) + (1 << (x + 1)), (1 << (x + 2)) - 1)]
            cnt = 0
            for aa, bb in tmp:
                j1 = j2 = n - 1
                res = 0
                for ii in range(n):
                    while j2 >= 0 and nums[j2] & mask > bb - (nums[ii] & mask):
                        j2 -= 1
                    while j1 >= 0 and nums[j1] & mask >= aa - (nums[ii] & mask):
                        j1 -= 1
                    cur = j2 - j1
                    if j1 + 1 <= ii <= j2:
                        cur -= 1
                    res += cur
                assert res % 2 == 0
                cnt += res // 2
            if cnt % 2:
                ans |= 1 << x
        ac.st(ans)
        return

    @staticmethod
    def arc_092b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/arc092/tasks/arc092_b
        tag: bit_operation|contribution_method|classical|two_pointers
        """
        max_bit = 30
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        ans = 0
        for x in range(max_bit):
            one, zero = [], []
            for num in a:
                if (num >> x) & 1:
                    one.append(num)
                else:
                    zero.append(num)
            a = zero + one

            one = []
            zero = []
            for num in b:
                if (num >> x) & 1:
                    one.append(num)
                else:
                    zero.append(num)
            b = zero + one

            mask = (1 << (x + 1)) - 1

            tmp = [(1 << x, (1 << (x + 1)) - 1), ((1 << x) + (1 << (x + 1)), (1 << (x + 2)) - 1)]
            cnt = 0
            for aa, bb in tmp:
                j1 = j2 = n - 1
                for ii in range(n):
                    while j2 >= 0 and b[j2] & mask > bb - (a[ii] & mask):
                        j2 -= 1
                    while j1 >= 0 and b[j1] & mask >= aa - (a[ii] & mask):
                        j1 -= 1
                    cur = j2 - j1
                    cnt += cur
            if cnt % 2:
                ans |= 1 << x
        ac.st(ans)
        return

    @staticmethod
    def lg_u360500(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/U360500
        tag: bit_operation|contribution_method|classical|two_pointers|sorted_list|brain_teaser
        """

        max_bit = 25
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums = ac.accumulate(nums)
        n += 1
        ans = c = 0
        for x in range(max_bit):
            a0, a1 = [], []
            sum_idx = 0
            for idx, num in enumerate(nums):
                if (num >> x) & 1:
                    a1.append(num)
                else:
                    a0.append(num)
                    sum_idx += idx
            n0, n1 = len(a0), len(a1)
            if c ^ (n0 * n1 % 2):
                ans |= 1 << x
            c ^= (n0 * (n0 - 1) // 2 - sum_idx) % 2
            nums = a0 + a1
        ac.st(ans)
        return

    @staticmethod
    def lc_3145(queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-products-of-elements-of-big-array/
        tag: bit_operation|data_range|classical|inclusion_exclusion|counter
        """
        m = 60
        count = [1] + [(1 << (i - 1)) * i + 1 for i in
                       range(1, m)]  # count[i]=sum(num.bit_count() for num in range((1<<i)|1))

        def check(num):
            ceil = pre = bit = 0
            for x in range(m - 1, -1, -1):
                if pre + count[x] + (bit << x) <= num:
                    ceil |= 1 << x
                    pre += count[x] + (bit << x)
                    bit += 1
            res = []
            ceil += 1
            for x in range(m):
                c = (ceil // (1 << (x + 1))) * (1 << x)
                if ceil % (1 << (x + 1)) > (1 << x):
                    c += ceil % (1 << (x + 1)) - (1 << x)
                res.append(c)

            diff = num - sum(res)
            if diff:
                for x in range(m):
                    if (ceil >> x) & 1:
                        res[x] += 1
                        diff -= 1
                        if not diff:
                            break
            return sum([j * res[j] for j in range(m)])

        ans = []
        for a, b, mod in queries:
            ans.append(pow(2, check(b + 1) - check(a) if a else check(b + 1), mod))
        return ans

    @staticmethod
    def cf_1557c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1557/C
        tag: bit_operation|brain_teaser|brute_force|observation
        """
        mod = 10 ** 9 + 7
        cb = Combinatorics(2 * 10 ** 5 + 10, mod)
        p = [0] * (2 * 10 ** 5 + 1)
        p[0] = 1
        for i in range(1, 2 * 10 ** 5 + 1):
            p[i] = p[i - 1] * 2 % mod
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            ans = 0
            pre = 1
            even = sum(cb.comb(n, x) for x in range(0, n, 2) if x < n)

            for i in range(k - 1, -1, -1):
                if n % 2 == 0:  # >
                    ans += pre * pow(p[i], n, mod)
                if n % 2:  # =
                    pre *= (even + 1)
                else:
                    pre *= even
                pre %= mod
                ans %= mod
            ans += pre
            ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def cf_1720d1(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1720/D1
        tag: linear_dp|data_range|observation|bit_operation
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            dp = [1] * n
            for i in range(1, n):
                for j in range(max(0, i - 400), i):
                    if (nums[j] ^ i) < (nums[i] ^ j):
                        dp[i] = max(dp[i], dp[j] + 1)
            ac.st(max(dp))
        return

    @staticmethod
    def cf_1715d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1715/D
        tag: greedy|construction|bit_operation
        """
        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        ans = [(1 << 30) - 1] * n
        for _ in range(q):
            x, y, num = ac.read_list_ints_minus_one()
            num += 1
            ans[x] &= num
            ans[y] &= num
            dct[x].append(y)
            dct[y].append(x)
        for i in range(n):
            if i not in dct[i]:
                v = ans[i]
                for j in dct[i]:
                    v &= ans[j]
                ans[i] ^= v
        ac.lst(ans)
        return

    @staticmethod
    def cf_1416c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1416/C
        tag: bit_operation|divide_and_conquer|reverse_pair
        """
        ac.read_int()
        nums = ac.read_list_ints()

        res = pre = 0
        arr = [nums[:]]
        for i in range(31, -1, -1):
            cur_one = 0
            cur_zero = 0
            nex = []
            for ls in arr:
                one = []
                zero = []
                for num in ls:
                    if (num >> i) & 1:
                        one.append(num)
                        cur_one += len(zero)
                    else:
                        zero.append(num)
                        cur_zero += len(one)
                if len(one) > 1:
                    nex.append(one)
                if len(zero) > 1:
                    nex.append(zero)
            arr = nex
            if cur_zero <= cur_one:
                pre += cur_zero
                continue
            pre += cur_one
            res |= 1 << i
        ac.lst([pre, res])
        return

    @staticmethod
    def cf_2036f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2036/problem/F
        tag: bit_property|bit_operation|inclusion_exclusion|classical
        """
        assert (1 << 64) >= 10 ** 18
        for _ in range(ac.read_int()):
            ll, rr, i, k = ac.read_list_ints()

            def check(x):
                res = BitOperation().sum_xor(x)
                cur = 0
                if x >= k:
                    ceil = 0
                    for j in range(64, i - 1, -1):
                        if ceil | (1 << j) | k <= x:
                            ceil |= (1 << j)
                    cur = (BitOperation().sum_xor(ceil >> i)) << i
                    if not (ceil >> i) & 1:
                        cur |= k
                return res ^ cur

            ans = check(rr) ^ check(ll - 1)
            ac.st(ans)
        return

    @staticmethod
    def cf_2035c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2035/problem/C
        tag: construction|bit_operation
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n == 6:
                ac.st(7)
                ac.lst([1, 2, 4, 6, 5, 3])
            elif n % 2:
                ac.st(n)
                ac.lst(list(range(2, n - 1)) + [1, n - 1, n])
            else:
                x = n.bit_length() - 1
                lst = [3, 1, 2 ** x - 2, 2 ** x - 1, n]
                ans = [x for x in range(1, n + 1) if x not in lst] + lst
                ac.st(2 ** (x + 1) - 1)
                ac.lst(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/MEXXOR?tab=statement
        tag: bit_operation|brain_teaser
        """
        for _ in range(ac.read_int()):
            ll, rr, xx = ac.read_list_ints()
            ans = 0
            for i in range(29, -1, -1):
                mask = xx & ~((1 << i) - 1)
                mn = ans ^ mask
                mx = mn + (1 << i) - 1
                if ll <= mn <= mx <= rr:
                    ans |= 1 << i
            ac.st(ans)
        return

    @staticmethod
    def cf_2020c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2020/problem/C
        tag: bit_operation|construction|observation
        """
        for _ in range(ac.read_int()):
            b, c, d = ac.read_list_ints()
            a = b ^ d
            if (a | b) - (a & c) == d:
                ac.st(a)
            else:
                ac.st(-1)
        return

    @staticmethod
    def lc_3141(nums: List[int], m: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-hamming-distances/
        tag: bfs|brain_teaser|build_graph|bit_operation
        """
        mask = (1 << m) - 1
        visit = [m] * (1 << m)
        stack = nums[:]
        for num in nums:
            visit[num] = 0
        while stack:
            nex = []
            for num in stack:
                for i in range(m):
                    x = num ^ (1 << i)
                    if visit[x] > visit[num] + 1:
                        visit[x] = visit[num] + 1
                        nex.append(x)
            stack = nex
        ans = [m - visit[num ^ mask] for num in nums]
        return ans
