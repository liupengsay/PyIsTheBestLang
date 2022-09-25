
import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq

def get_all_factor(num):
    factor = set()
    for i in range(1, int(math.sqrt(num))+1):
        if num % i == 0:
            factor.add(i)
            factor.add(num//i)
    return sorted(list(factor))

def divPrime(num):
    lt = []
    while num != 1:
        for i in range(2, int(num+1)):
            if num % i == 0:  # i是num的一个质因数
                lt.append(i)
                num = num / i # 将num除以i，剩下的部分继续分解
                break
    return lt


class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k == 1:
            return n*(n-1)//2
        ans = 0
        pre = defaultdict(int)
        for i, num in enumerate(nums):
            a = num//math.gcd(num, k)
            ans += pre[a]
            for f in get_all_factor(num):
                pre[f] += 1
        return ans

def test_solution():
    assert Solution().numberOfGoodPaths(vals = [1,3,2,1,3], edges = [[0,1],[0,2],[2,3],[2,4]]) == 6
    return



if __name__ == '__main__':
    test_solution()
