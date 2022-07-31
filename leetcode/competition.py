
from functools import lru_cache
import math
math.gcd()
@lru_cache(None)
def get_prime_factor(num):
    if num == 1:
        return []
    ans = set()
    for i in range(2, int(math.sqrt(num))+2):
        if num % i == 0:
            num //= i
            ans.add(i)
            ans = ans.union(get_prime_factor(num))
            return ans
    ans.add(num)
    return ans


from collections import defaultdict


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1]*n
        self.part = n

    def find(self, x):
        if x != self.root[x]:
            root_x = self.find(self.root[x])
            self.root[x] = root_x
            return root_x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.part -= 1
        return


class Solution:
    def largestComponentSize(self, nums) -> int:
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            for prime_factor in get_prime_factor(num):
                dct[prime_factor].append(i)
        n = len(nums)
        uf = UnionFind(n)
        for group in dct:
            m = len(dct[group])
            for j in range(1, m):
                uf.union(dct[group][j-1], dct[group][j])
        return max(uf.size)



print(get_prime_factor(3))
print(get_prime_factor(6))