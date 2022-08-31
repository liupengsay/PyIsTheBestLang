
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
MOD = 10**9 + 7


class Solution:
    def maxProfit(self, inventory: List[int], orders: int) -> int:
        n = len(inventory)

        def check(ceil):
            cnt = 0
            for num in inventory:
                if num > ceil:
                    cnt += num - ceil
            return cnt >= orders

        low = 0
        high = max(inventory)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        rest = high if check(high) else low
        count = 0
        ans = 0
        for num in inventory:
            if num > rest:
                cur = num - rest
                count += cur
                ans += cur * (2 * rest + cur + 1) // 2
                ans %= MOD
        ans += (orders - count) * rest
        ans %= MOD
        return ans


print(Solution().bestTeamScore([4, 5, 6, 5], ages=[2, 1, 2, 1]))
#
# assert Solution().maxProductPath(
#     [[-1, -2, -3], [-2, -3, -3], [-3, -3, -2]]) == -1
# assert Solution().maxProductPath([[1, -2, 1], [1, -2, 1], [3, -4, 1]]) == 8
# assert Solution().maxProductPath([[1, 3], [0, -4]]) == 0
# assert Solution().maxProductPath(
#     [[1, 4, 4, 0], [-2, 0, 0, 1], [1, -1, 1, 1]]) == 2
