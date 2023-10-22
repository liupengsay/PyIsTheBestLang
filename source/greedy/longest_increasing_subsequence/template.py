import bisect
from bisect import bisect_left
from collections import deque, defaultdict
from typing import List


class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # longest strictly increasing subsequence
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # longest non-decreasing subsequence
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # longest strictly decreasing subsequence
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # longest strictly non-increasing subsequence
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class LcsLis:
    def __init__(self):
        return

    def length_of_lcs(self, s1, s2) -> int:
        """compute lcs with lis"""
        # O(n**2)
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        return self.length_of_lis(nums)

    @staticmethod
    def length_of_lis(nums: List[int]) -> int:
        # greedy and binary search to check lis
        stack = []
        for x in nums:
            idx = bisect_left(stack, x)
            if idx < len(stack):
                stack[idx] = x
            else:
                stack.append(x)
        # length of lis
        return len(stack)

    def index_of_lcs(self, s1, s2) -> List[int]:
        # greedy and binary search to check lis output the index
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])
        # return the index of lcs in s2
        res = self.minimum_lexicographical_order_of_lis(nums)
        return res

    @staticmethod
    def minimum_lexicographical_order_of_lis(nums: List[int]) -> List[int]:
        """template of minimum lexicographical order lis"""
        # greedy and binary search to check lis output the index
        if not nums:
            return []
        n = len(nums)
        tops = [nums[0]]
        piles = [0] * n
        piles[0] = 0

        for i in range(1, n):
            if nums[i] > tops[-1]:
                piles[i] = len(tops)
                tops.append(nums[i])
            else:
                j = bisect.bisect_left(tops, nums[i])
                piles[i] = j
                tops[j] = nums[i]

        lis = []
        j = len(tops) - 1
        for i in range(n - 1, -1, -1):
            if piles[i] == j:
                lis.append(nums[i])
                j -= 1
        lis.reverse()
        return lis

    @staticmethod
    def length_and_max_sum_of_lis(nums):
        """the maximum sum of lis with maximum length"""
        # which can be extended to non-decreasing non-increasing minimum sum and so on
        dp = []
        q = []
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                q[length].append([num, num])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    q[length - 1].popleft()
                cur = q[length - 1][0][1] + num
                while q[length] and q[length][-1][1] <= cur:
                    q[length].pop()
                q[length].append([num, cur])
        return q[-1][0][1]

    @staticmethod
    def length_and_cnt_of_lis(nums):
        """template to finding the number of LIS"""
        # O(nlogn)
        dp = []  # LIS array
        s = []  # index if length and value is sum
        q = []  # index if length and value is [num, cnt]
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                q[length].append([num, s[length - 1]])
        return s[-1]

    @staticmethod
    def length_and_cnt_of_lcs(s1, s2, mod=10 ** 9 + 7):
        """template of number of lcs calculated by lis"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        dp = []
        s = []
        q = []
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                s[length] %= mod
                q[length].append([num, s[length - 1]])
        return len(dp), s[-1]
