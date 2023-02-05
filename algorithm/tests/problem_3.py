import unittest

from typing import List
from collections import defaultdict, Counter

import heapq
import heapq
import unittest
from collections import defaultdict, Counter
from typing import List


class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        heapq.heapify(basket1)
        heapq.heapify(basket2)
        x = min(basket1)
        lst = 0
        change1 = []
        change2 = []
        while basket1 and basket2:
            if basket1[0]==basket2[0]:
                heapq.heappop(basket2)
                heapq.heappop(basket1)
                continue
            a, b = heapq.heappop(basket1), heapq.heappop(basket2)
            if a < b:
                if not (basket1 and basket1[0]==a):
                    return -1
                lst += 1
                change1.append(a)
                change2.append(b)
                heapq.heappop(basket1)
                heapq.heappush(basket1, b)
            else:
                if not (basket2 and basket2[0]==b):
                    return -1
                lst += 1
                heapq.heappop(basket2)
                heapq.heappush(basket2, a)
        ans = lst*min(x, y)
        if lst%2 == 0:
            return
        return ans


class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        cnt1 = Counter(basket1)
        cnt2 = Counter(basket2)
        cnt = cnt1 + cnt2
        target = defaultdict(int)
        for w in cnt:
            if cnt[w]%2:
                return -1
            target[w] = cnt[w]//2

        x, y = min(basket1), min(basket2)

        ans1 = ans2 = 0
        z = 0
        for w in cnt1:
            x = cnt1[w]-target[w]
            if x > 0:
                z += x
        if z%2 == 1:
            return min(x,y)*z*2
        return min(w for w in cnt1 if not w)*z



class Solution:
    def minCapability(self, nums: List[int], k: int) -> int:
        n = len(nums)

        def check(x):

            dp = [0]*(n+1)
            for i in range(n):
                dp[i+1] = dp[i]
                if nums[i] <= x and dp[i-1]+1>dp[i+1]:
                    dp[i+1] = dp[i-1] + 1
            return dp[-1] >= k

        low = 0
        high = max(nums)
        while low < high-1:
            mid = low+(high-low)//2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        nums
        assert Solution().isPossible(
            4, [[1, 2], [2, 3], [2, 4], [3, 4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
