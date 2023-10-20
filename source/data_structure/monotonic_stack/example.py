import random
import unittest

from data_structure.monotonic_stack.template import MonotonicStack


class TestGeneral(unittest.TestCase):

    def test_monotonic_stack(self):
        n = 1000
        nums = [random.randint(0, n) for _ in range(n)]
        ms = MonotonicStack(nums)
        for i in range(n):

            # 上一个最值
            pre_bigger = pre_bigger_equal = pre_smaller = pre_smaller_equal = -1
            for j in range(i - 1, -1, -1):
                if nums[j] > nums[i]:
                    pre_bigger = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] >= nums[i]:
                    pre_bigger_equal = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] < nums[i]:
                    pre_smaller = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] <= nums[i]:
                    pre_smaller_equal = j
                    break
            assert pre_bigger == ms.pre_bigger[i]
            assert pre_bigger_equal == ms.pre_bigger_equal[i]
            assert pre_smaller == ms.pre_smaller[i]
            assert pre_smaller_equal == ms.pre_smaller_equal[i]

            # 下一个最值
            post_bigger = post_bigger_equal = post_smaller = post_smaller_equal = - 1
            for j in range(i + 1, n):
                if nums[j] > nums[i]:
                    post_bigger = j
                    break
            for j in range(i + 1, n):
                if nums[j] >= nums[i]:
                    post_bigger_equal = j
                    break
            for j in range(i + 1, n):
                if nums[j] < nums[i]:
                    post_smaller = j
                    break
            for j in range(i + 1, n):
                if nums[j] <= nums[i]:
                    post_smaller_equal = j
                    break
            assert post_bigger == ms.post_bigger[i]
            assert post_bigger_equal == ms.post_bigger_equal[i]
            assert post_smaller == ms.post_smaller[i]
            assert post_smaller_equal == ms.post_smaller_equal[i]

        return


if __name__ == '__main__':
    unittest.main()
