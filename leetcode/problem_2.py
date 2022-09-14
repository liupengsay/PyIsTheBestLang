
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict


from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict





def test_solution():
    assert Solution().largestMerge("cabaa", word2="bcaaa") == "cbcabaaaaa"
    assert Solution().largestMerge("ab", word2="abcd") == "abcdab"
    assert Solution().largestMerge("ab", word2="aba") == "ababa"

    return


if __name__ == '__main__':
    test_solution()
