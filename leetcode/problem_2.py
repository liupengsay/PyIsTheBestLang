
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
from itertools import combinations, permutations
from sortedcontainers import SortedDict
from decimal import Decimal

from collections import deque



# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        lst = []
        cur = head
        pre = 0
        while cur:
            if cur.val == 0:
                if pre:
                    lst.append(pre)
                pre = 0
            else:
                pre += cur.val
            cur = cur.next
        if not lst:
            return
        cur = head
        i = 0
        n = len(lst)
        while cur:
            cur.val = lst[i]
            i += 1
            if i == n:
                cur.next = None
            cur = cur.next
        return head


def test_solution():
    m = 976
    assert Solution().ballGame(m, p) == 1
    return


if __name__ == '__main__':
    test_solution()
