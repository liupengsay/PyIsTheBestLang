

import bisect

from typing import List

import math
from collections import defaultdict, Counter
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet


class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        n = len(s)

        def dfs(st, i):
            nonlocal ans
            if i == n:
                if st and st not in pre:
                    pre.add(st)
                    if len(pre) > ans:
                        ans = len(pre)
                    pre.discard(st)
                return
            if st and st not in pre:
                pre.add(st)
                dfs(s[i], i + 1)
                pre.discard(st)
            dfs(st + s[i], i + 1)
            return

        ans = 0
        pre = set()
        dfs('', 0)
        return ans


assert Solution().maxUniqueSplit("ababccc") == 5
assert Solution().maxUniqueSplit("aba") == 2
assert Solution().maxUniqueSplit("aa") == 1
assert Solution().maxUniqueSplit("aaaa") == 2
assert Solution().maxUniqueSplit("abvghjkmloiuyhng") == 15
