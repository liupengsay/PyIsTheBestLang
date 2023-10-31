import random
import unittest

from src.data_structure.priority_queue.template import PriorityQueue


class TestGeneral(unittest.TestCase):

    def test_priority_queue(self):
        pq = PriorityQueue()

        for _ in range(10):
            n = random.randint(100, 1000)
            nums = [random.randint(1, n) for _ in range(n)]
            k = random.randint(1, n)
            ans = pq.sliding_window(nums, k, "max")
            for i in range(n - k + 1):
                assert ans[i] == max(nums[i:i + k])

            ans = pq.sliding_window(nums, k, "min")
            for i in range(n - k + 1):
                assert ans[i] == min(nums[i:i + k])
        return


if __name__ == '__main__':
    unittest.main()
