"""
Algorithm：hash|contribution_method|matrix_hash|tree_hash|string_hash|prefix_hash|suffix_hash|hash_crush
Description：prefix_suffix|counter|index|prefix_sum

====================================LeetCode====================================
2143（https://leetcode.cn/problems/choose-numbers-from-two-arrays-in-range/）prefix_sum|hash|counter
17（https://leetcode.cn/problems/find-longest-subarray-lcci/）prefix_sum|hash
1590（https://leetcode.cn/problems/make-sum-divisible-by-p/）prefix_sum|hash|mod|counter
2588（https://leetcode.cn/problems/count-the-number-of-beautiful-subarrays/）prefix_sum|hash|counter
02（https://leetcode.cn/contest/hhrc2022/problems/0Wx4Pc/）prefix_sum|hash|brain_teaser|greedy
03（https://leetcode.cn/contest/hhrc2022/problems/VAc7h3/）tree_hash
2031（https://leetcode.cn/problems/count-subarrays-with-more-ones-than-zeros/）prefix_sum|hash|counter
2025（https://leetcode.cn/problems/maximum-number-of-ways-to-partition-an-array/description/）hash|contribution_method|counter
895（https://leetcode.cn/problems/maximum-frequency-stack/description/）hash|stack
1658（https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/description/）prefix_sum|hash|brain_teaser|greedy
2227（https://leetcode.cn/problems/encrypt-and-decrypt-strings/）brain_teaser|reverse_thinking

===================================CodeForces===================================

=====================================LuoGu======================================
P2697（https://www.luogu.com.cn/problem/P2697）hash|prefix_sum
P1114（https://www.luogu.com.cn/problem/P1114）hash|prefix_sum
P4889（https://www.luogu.com.cn/problem/P4889）math|hash|counter
P6273（https://www.luogu.com.cn/problem/P6273）hash|prefix|counter
P8630（https://www.luogu.com.cn/problem/P8630）hash|counter|permutation|brute_force

====================================AtCoder=====================================
D - Snuke's Coloring（https://atcoder.jp/contests/abc045/tasks/arc061_b）hash|inclusion_exclusion|counter


=====================================AcWing=====================================
137（https://www.acwing.com/problem/content/139/）matrix_hash

"""
import random
from collections import defaultdict, Counter
from itertools import accumulate
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2143(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/choose-numbers-from-two-arrays-in-range/
        tag: prefix_sum|hash|counter
        """
        # hashcounterimplementionlinear_dp 转移
        n = len(nums1)
        mod = 10 ** 9 + 7
        pre = defaultdict(int)
        pre[-nums1[0]] += 1
        pre[nums2[0]] += 1
        ans = pre[0]
        for i in range(1, n):
            cur = defaultdict(int)
            cur[-nums1[i]] += 1
            cur[nums2[i]] += 1
            for p in pre:
                cur[p - nums1[i]] += pre[p]
                cur[p + nums2[i]] += pre[p]
            ans += cur[0]
            ans %= mod
            pre = cur
        return ans

    @staticmethod
    def lc_1658(nums: List[int], x: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/description/
        tag: prefix_sum|hash|brain_teaser|greedy
        """
        # prefix_sumhash，|brain_teasergreedy
        pre = {0: -1}
        cur = 0
        n = len(nums)
        s = sum(nums)  # 先求和为 s-x 的最长子数组
        ans = -1 if s != x else 0
        for i, w in enumerate(nums):
            cur += w
            if cur - (s - x) in pre and i - pre[cur - (s - x)] > ans:  # 要求非空
                ans = i - pre[cur - (s - x)]
            if cur not in pre:
                pre[cur] = i
        return n - ans if ans > -1 else ans

    @staticmethod
    def lc_2025(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-ways-to-partition-an-array/description/
        tag: hash|contribution_method|counter
        """

        # 厘清边界hashcontribution_methodcounter
        n = len(nums)
        ans = 0
        pre = list(accumulate(nums, initial=0))
        for i in range(1, n):
            if pre[i] == pre[-1] - pre[i]:
                ans += 1

        # 左-右
        cnt = [0] * n
        post = defaultdict(int)
        for i in range(n - 2, -1, -1):
            b = pre[-1] - pre[i + 1]
            a = pre[i + 1]
            post[a - b] += 1
            # 作为左边
            cnt[i] += post[nums[i] - k]

        # 右-左
        dct = defaultdict(int)
        for i in range(1, n):
            b = pre[-1] - pre[i]
            a = pre[i]
            dct[a - b] += 1
            # 作为右边
            cnt[i] += dct[k - nums[i]]

        return max(ans, max(cnt))

    @staticmethod
    def abc_45d(ac=FastIO()):
        # hash容斥counter
        h, w, n = ac.read_list_ints()
        cnt = [0] * 10
        dct = defaultdict(int)
        for _ in range(n):
            a, b = ac.read_list_ints()
            for x in range(3):
                for y in range(3):
                    if 3 <= x + a <= h and 3 <= y + b <= w:
                        dct[(x + a, y + b)] += 1
        for k in dct:
            cnt[dct[k]] += 1

        cnt[0] = (h - 2) * (w - 2) - sum(cnt[1:])
        for a in cnt:
            ac.st(a)
        return

    @staticmethod
    def ac_137(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/139/
        tag: matrix_hash
        """

        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        def compute(ls):
            res1 = 0
            for num in ls:
                res1 *= p1
                res1 += num
                res1 %= mod1
            res2 = 0
            for num in ls:
                res2 *= p2
                res2 += num
                res2 %= mod2
            return res1, res2

        def check():
            res = []
            for ii in range(6):
                cu = tuple(lst[ii:] + lst[:ii])
                res.append(compute(cu))
                cu = tuple(lst[:ii + 1][::-1] + lst[ii + 1:][::-1])
                res.append(compute(cu))
            return res

        n = ac.read_int()
        pre = set()
        ans = False
        for _ in range(n):
            if ans:
                break
            lst = ac.read_list_ints()
            now = check()
            if any(cur in pre for cur in now):
                ans = True
                break
            for cur in now:
                pre.add(cur)

        if ans:
            ac.st("Twin snowflakes found.")
        else:
            ac.st("No two snowflakes are alike.")
        return

    @staticmethod
    def lg_p4889(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4889
        tag: math|hash|counter
        """
        # brute_forcecounter
        n, m = ac.read_list_ints()
        height = ac.read_list_ints()
        cnt = defaultdict(int)
        ans = 0
        for i in range(n):
            # hj - hi = j - i
            ans += cnt[height[i] - i]
            cnt[height[i] - i] += 1

        cnt = defaultdict(int)
        for i in range(n):
            # hi - hj = j - i
            ans += cnt[height[i] + i]
            # hj + hi = j - i
            ans += cnt[i - height[i]]
            cnt[height[i] + i] += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p6273(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6273
        tag: hash|prefix|counter
        """
        # hash前缀counter
        ac.read_int()
        s = ac.read_str()
        # 选择最少出现的字符作为减数
        ct = Counter(s)
        st = list(ct.keys())
        ind = {w: i for i, w in enumerate(st)}
        m = len(ind)
        x = 0
        for i in range(1, m):
            if ct[st[i]] < ct[st[x]]:
                x = i
        # 记录状态
        cnt = [0] * m
        pre = defaultdict(int)
        pre[tuple(cnt)] = 1
        ans = 0
        mod = 10 ** 9 + 7
        for w in s:
            if w == st[x]:
                # 其余所有字符减 1
                for i in range(m):
                    if i != ind[w]:
                        cnt[i] -= 1
            else:
                # 减数字符| 1
                cnt[ind[w]] += 1
            tp = tuple(cnt)
            # sa-ta = sb-tb 则有 sa-sb = ta-tb 因此这样counter
            ans += pre[tp]
            pre[tp] += 1
            ans %= mod
        ac.st(ans)
        return


class LC895:
    # hash与stack的结合应用题
    def __init__(self):
        self.freq = defaultdict(list)
        self.dct = defaultdict(int)
        self.ceil = 0

    def push(self, val: int) -> None:
        self.dct[val] += 1
        self.freq[self.dct[val]].append(val)
        if self.dct[val] > self.ceil:
            self.ceil = self.dct[val]
        return

    def pop(self) -> int:
        val = self.freq[self.ceil].pop()
        self.dct[val] -= 1
        if not self.freq[self.ceil]:
            self.ceil -= 1
        return val
