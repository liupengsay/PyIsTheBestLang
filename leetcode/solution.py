

from typing import List


import bisect

# Definition for polynomial singly-linked list.
# class PolyNode:
#     def __init__(self, x=0, y=0, next=None):
#         self.coefficient = x
#         self.power = y
#         self.next = next


from collections import defaultdict


class Solution:
    def shortestPalindrome(self, s: str) -> str:
