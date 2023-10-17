import unittest
from typing import List, Optional


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6914_1(head: Optional[ListNode]) -> Optional[ListNode]:
        # 模板：链表与整数相乘
        lno = ListNodeOperation()
        lst = lno.node_to_lst(head)[::-1]

        nums = []
        x = 0
        for num in lst:
            x += num * 2
            nums.append(x % 10)
            x = 1 if x >= 10 else 0
        if x:
            nums.append(x)

        nums.reverse()
        return lno.lst_to_node(nums)

    @staticmethod
    def lc_6914_2(head: Optional[ListNode]) -> Optional[ListNode]:
        lno = ListNodeOperation()
        num = lno.node_to_num(head)*2
        return lno.num_to_node(num)

