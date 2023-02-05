from __future__ import division

import unittest


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        ans = None
        while head:
            tmp = head.next
            head.next = ans
            ans= head
            head = tmp
        return ans


class TestGeneral(unittest.TestCase):

    def test_solution(self):
        assert Solution().getMaximumConsecutive([1, 3]) == 2

        return


if __name__ == '__main__':
    unittest.main()
