

import bisect

from typing import List

import math
from collections import defaultdict, Counter
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet


class Solution:
    def reorderSpaces(self, text: str) -> str:
        cnt = text.count(' ')
        lst = text.split(' ')
        lst = [ls for ls in lst if ls]
        n = len(lst)
        if n == 1:
            return lst[0] + ' '*cnt
        ans = (' ' * (cnt // (n - 1))).join(lst)
        ans += ' ' * (cnt % (n - 1))
        return ans


assert Solution().reorderSpaces(
    "  this   is  a sentence ") == "this   is   a   sentence"
assert Solution().reorderSpaces(
    " practice   makes   perfect") == "practice   makes   perfect "
assert Solution().reorderSpaces("hello   world") == "hello   world"
assert Solution().reorderSpaces(
    "  walks  udp package   into  bar a") == "walks  udp  package  into  bar  a "
assert Solution().reorderSpaces("a") == "a"
assert Solution().reorderSpaces("  aweefe  ") == "aweefe    "