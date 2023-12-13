"""
Algorithm：list_node|linked_list
Description：

====================================LeetCode====================================
6914（https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/）linked_list

"""
from typing import Optional

from src.data_structure.list_node.template import ListNode, ListNodeOperation


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6914_1(head: Optional[ListNode]) -> Optional[ListNode]:
        """
        url: https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/
        tag: linked_list
        """
        # linked_list|与整数相乘
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
        """
        url: https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/
        tag: linked_list
        """
        lno = ListNodeOperation()
        num = lno.node_to_num(head) * 2
        return lno.num_to_node(num)
