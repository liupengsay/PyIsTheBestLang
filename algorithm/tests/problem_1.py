

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from operator import xor, mul, add
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        ans = ListNode(-1)
        pre = ans
        while list1 and list2:
            if list1.val >= list2:
                tmp = list2.next
                list2.next = None
                pre.next = list2
                pre = pre.next
                list2 = tmp
            else:
                tmp = list1.next
                list1.next = None
                pre.next = list1
                pre = pre.next
                list1 = tmp
        while list1:
            tmp = list2.next
            list2.next = None
            pre.next = list2
            pre = pre.next
            list2 = tmp
        while list2:
            tmp = list1.next
            list1.next = None
            pre.next = list1
            pre = pre.next
            list1 = tmp
        return ans.next




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5,1,1,2,0,0]) == [0,0,1,1,2,5]
        return


if __name__ == '__main__':
    unittest.main()
