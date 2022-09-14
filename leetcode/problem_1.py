

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


from functools import lru_cache




def test_solution():
    assert Solution().earliestAndLatest(5, 1, 5) == [1, 1]
    assert Solution().earliestAndLatest(27, 26, 27) == [5, 5]
    assert Solution().earliestAndLatest(27, 1, 26) == [2, 2]
    return


if __name__ == '__main__':
    test_solution()
