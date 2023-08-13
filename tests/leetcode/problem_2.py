import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

sys.set_int_max_str_digits(0)

sys.set_int_max_str_digits(0) # 特别重要，防止出现整数超限，默认不超过4300位数


class ListNodeOperation:
    def __init__(self):
        return

    @staticmethod
    def node_to_num(node: ListNode) -> int:
        num = 0
        while node:
            num = num * 10 + node.val
            node = node.next
        return num

    @staticmethod
    def num_to_node(num: int) -> ListNode:
        node = ListNode(-1)
        pre = node
        for x in str(num):
            pre.next = ListNode(int(x))
            pre = pre.next
        return node.next


class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        lno = ListNodeOperation()
        num = lno.node_to_num(head)*2
        return lno.num_to_node(num)


assert Solution()
