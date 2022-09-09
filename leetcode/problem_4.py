
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict


dp = [[1], [1,1,1]]
for i in range(2, 31):
    lst = []
    for j in range(len(dp[-1])-1):
        lst.append(dp[-1][j]+dp[-1][j+1])
    dp.append([1]+lst+[1])

cnt = [sum(d) for d in dp]

class Solution:
    def minimumBoxes(self, n: int) -> int:
        for i in range(31):
            n -= sum(dp[i])
            if n < 0:
                return sum(dp[i-1])

@lru_cache(None)
def dfs(n):
    if n <= 2:
        return n
    k = math.floor(math.sqrt(2*n+1/4)-1/2)
    cur = k*(k+1)//2
    return n+dfs(k*(k-1)//2+max(0, n-cur-1))


print(dfs(5))
class Solution:
    def minimumBoxes(self, n: int) -> int:

        def check(num):
            return dfs(num) >= n

        low = 1
        high = n
        while low < high-1:
            mid = low+(high-low)//2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high



def test_solution():
    assert Solution().minimumBoxes(15) == 9
    assert Solution().minimumBoxes(10) == 6
    assert Solution().minimumBoxes(3) == 3
    assert Solution().minimumBoxes(4) == 3
    return


if __name__ == '__main__':
    test_solution()
