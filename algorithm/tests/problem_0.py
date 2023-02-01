import copy
import random
import heapq
import math
import sys
import bisect
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from types import GeneratorType
from functools import cmp_to_key
from heapq import nlargest
inf = float("inf")
sys.setrecursionlimit(10000000)


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return int(self._read())

    def read_ints(self):
        return map(int, self._read().split())

    def read_floats(self):
        return map(float, self._read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self._read().split())

    def read_list_ints(self):
        return list(map(int, self._read().split()))

    def read_list_floats(self):
        return list(map(float, self._read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self._read().split()))

    def read_str(self):
        return self._read()

    def read_list_strs(self):
        return self._read().split()

    def read_list_str(self):
        return list(self._read())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        # 四舍五入保留d位小数 '{:.df}'.format(ans)
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to
        return wrappedfunc



def main(ac=FastIO()):
    n = ac.read_int()
    grid = [ac.read_list_str() for _ in range(n)]
    height = [ac.read_list_ints() for _ in range(n)]

    nums = []
    house = []
    start = [-1, -1]
    low = inf
    high = -inf
    uf = UnionFind(n * n)
    for i in range(n):
        for j in range(n):
            nums.append([i, j, height[i][j]])
            if grid[i][j] == "K":
                house.append(i*n+j)
                low = ac.min(low, height[i][j])
                high = ac.max(high, height[i][j])
                uf.size[i*n+j] = 1
            elif grid[i][j] == "P":
                start = [i, j]
                low = ac.min(low, height[i][j])
                high = ac.max(high, height[i][j])

    nums.sort(key=lambda x: x[2])


    def check(gap):
        j = 0
        for i in range(n*n):
            # if nums[i][2] > low:
            #     break
            # if nums[i][2] >= high-gap:
            while j<n*n and nums[j][2] <= nums[i][2]+gap:
                j += 1
            j -= 1
            if j>=i and nums[j][2]-nums[i][2] <= gap and nums[i][2]<=low and nums[j][2]>=high:
                a, b = nums[i][2], nums[j][2]
                uf1 = copy.deepcopy(uf)
                for x, y, _ in nums[i:j+1]:
                    for p, q in [[x-1,y],[x+1,y],[x,y+1],[x,y-1], [x-1,y-1],[x-1,y+1],[x+1,y-1],[x+1,y+1]]:
                        if 0<=p<n and 0<=q<n and a<=height[p][q] <= b:
                            uf1.union(x*n+y, p*n+q)
                if uf1.size[uf1.find(start[0]*n+start[1])] == len(house):
                    return True

        return False

    floor = high-low
    ceil = max(x for _,_, x in nums) - min(x for _,_, x in nums)
    while floor < ceil-1:
        mid = floor + (ceil-floor)//2
        if check(mid):
            ceil = mid
        else:
            floor = mid
    ans = floor if check(floor) else ceil
    ac.st(ans)
    return


main()
