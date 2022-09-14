import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import random


from collections import defaultdict


class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:

        def check(st):
            if len(st) == 1:
                return True
            if st[0] == "0":
                return False
            return int(st) <= 255

        def dfs(pre, st, i):
            if len(pre) > 4:
                return
            if st and not check(st):
                return
            if i == n:
                if st:
                    pre.append(st)
                if len(pre) == 4 and all(check(st) for st in pre) and len("".join(pre)) == n:
                    ans.append(".".join(pre))
                return
            if st:
                dfs(pre+[st], s[i], i + 1)
            dfs(pre, st + s[i], i + 1)
            return

        n = len(s)
        ans = []
        if n > 12:
            return ans
        dfs([], "", 0)
        return ans


def test_solution():
    assert Solution().minDistance([2, 3, 5, 12, 18], 2) == 9
    return


if __name__ == '__main__':
    test_solution()
