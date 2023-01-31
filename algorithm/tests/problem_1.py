

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
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """

        def reverse(node):
            res = None
            while node:
                tmp = node.next
                node.next = res
                res = node
                node = tmp
            return res


        def merge(node1, node2):
            res = ListNode(-1)
            node = res
            while node1 and node2:
                tmp = node1.next
                node1.next = None
                node.next = node1
                node = node.next
                node1 = tmp

                tmp = node2.next
                node2.next = None
                node.next = node2
                node = node.next
                node2 = tmp

            if node1:
                node.next = node1
            if node2:
                node.next = node2
            return res.next

        ans = ListNode(-1)
        ans.next = head
        fast = slow = ans
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        post = reverse(slow.next)
        slow.next = None
        pre = ans.next
        return merge(pre, post)



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
        return


if __name__ == '__main__':
    unittest.main()
