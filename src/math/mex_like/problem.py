"""
Algorithm：Minimum Excluded Element
Ability：brain storming like or g greedy
Reference：

====================================LeetCode====================================
330（https://leetcode.cn/problems/patching-array/）greedy|sort|implemention|mex
1798（https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/）greedy|sort|implemention|mex
2952（https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/）greedy|sort|implemention|mex

======================================Luogu=====================================
P9202（https://www.luogu.com.cn/problem/P9202）mex|operation
P9199（https://www.luogu.com.cn/problem/P9199）mex|operation

===================================CodeForces===================================
1905D（https://codeforces.com/contest/1905/problem/D）mex|contribution_method|diff_array|implemention|classical
1364C（https://codeforces.com/problemset/problem/1364/C）mex|construction
2021B（https://codeforces.com/contest/2021/problem/B）mex_like

===================================CodeChef===================================
1（https://www.codechef.com/problems/LIMITMEX）monotonic_stack|contribution_method|brain_teaser|classical

"""
from typing import List

from src.util.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_330(nums: List[int], n: int) -> int:
        """
        url: https://leetcode.cn/problems/patching-array/
        tag: greedy|sort|implemention|mex|classical
        """
        nums.sort()
        m = len(nums)
        i = 0
        mex = 1
        ans = 0
        while mex <= n:
            if i < m and nums[i] <= mex:
                mex += nums[i]
                i += 1
            else:
                ans += 1
                mex *= 2
        return ans

    @staticmethod
    def lc_2952(nums: List[int], n: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/
        tag: greedy|sort|implemention|mex
        """
        nums.sort()
        m = len(nums)
        i = 0
        mex = 1
        ans = 0
        while mex <= n:
            if i < m and nums[i] <= mex:
                mex += nums[i]
                i += 1
            else:
                ans += 1
                mex *= 2
        return ans

    @staticmethod
    def lc_1798(coins: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/
        tag: greedy|sort|implemention|mex
        """
        coins.sort()
        mex = 1
        for coin in coins:
            if coin <= mex:
                mex += coin
        return mex

    @staticmethod
    def cf_1905d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1905/problem/D
        tag：mex|contribution_method|diff_array|implemention|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            ind = [-1] * n
            for i in range(n):
                ind[nums[i]] = i

            pre = [-1] * n
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and nums[stack[-1]] > nums[i]:
                    pre[stack.pop()] = i
                stack.append(i)

            post = [-1] * n
            stack = []
            for i in range(n):
                while stack and nums[stack[-1]] > nums[i]:
                    post[stack.pop()] = i
                stack.append(i)

            diff = [0] * n
            pre_min = pre_max = ind[0]
            for i in range(1, n):
                x = ind[i]
                if post[x] != -1:
                    a = post[x]
                else:
                    a = pre_min
                if pre[x] != -1:
                    b = pre[x]
                else:
                    b = pre_max

                pre_min = min(pre_min, x)
                pre_max = max(pre_max, x)

                move1 = n - a
                move1 %= n
                cur_x = (x + move1) % n
                cur_b = (b + move1) % n

                length = cur_x - cur_b

                move2 = n - cur_x - 1
                if move1 + move2 <= n - 1:
                    diff[move1] += i * length
                    if move1 + move2 + 1 < n:
                        diff[move1 + move2 + 1] -= i * length
                else:
                    diff[move1] += i * length
                    xx = (move1 + move2) % n
                    diff[0] += i * length
                    if xx + 1 < n:
                        diff[xx + 1] -= i * length

            for i in range(1, n):
                diff[i] += diff[i - 1]
            ac.st(max(diff) + n)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/LIMITMEX
        tag: monotonic_stack|contribution_method|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
            pre = [0] * n  # initial can be 0 or -1 dependent on usage
            stack = []
            for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
                while stack and nums[stack[-1]] < nums[i]:  # can be < or > or <=  or >=  dependent on usage
                    post[stack.pop()] = i - 1  # can be i or i-1 dependent on usage
                if stack:  # which can be done only pre and post are no-repeat such as post bigger and pre not-bigger
                    pre[i] = stack[-1] + 1  # can be stack[-1] or stack[-1]-1 dependent on usage
                stack.append(i)
            ans = 0
            for i in range(n):
                ans += (nums[i] + 1) * (i - pre[i] + 1) * (post[i] - i + 1)

            post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
            pre = [0] * n  # initial can be 0 or -1 dependent on usage
            dct = dict()
            for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
                if nums[i] in dct:
                    post[dct[nums[i]]] = i - 1
                dct[nums[i]] = i  # can be i or i-1 dependent on usage
            for i in range(n):
                ans -= (i - pre[i] + 1) * (post[i] - i + 1)
            ac.st(ans)
        return
