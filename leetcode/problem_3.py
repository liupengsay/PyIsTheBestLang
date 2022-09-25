
import bisect
import itertools
import random
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations

class Solution:
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        cnt = sorted([list(item) for item in Counter(s).items()], key=lambda x: x[0], reverse=True)
        ans = ""
        while cnt:
            add = min(cnt[0][1], repeatLimit)
            ans += cnt[0][0]*add
            cnt[0][1] -= add
            if not cnt[0][1]:
                cnt.pop(0)
                continue
            else:
                if len(cnt) >= 2:
                    ans += cnt[1][0]
                    cnt[1][1] -= 1
                    if not cnt[1][1]:
                        cnt.pop(1)
                else:
                    break
        return ans















def test_solution():
    assert Solution().possiblyEquals("98u8v8v8v89u888u998v88u98v88u9v99u989v8u", "9v898u98v888v89v998u98v9v888u9v899v998u9") == False
    return


if __name__ == '__main__':
    test_solution()
