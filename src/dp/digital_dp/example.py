import unittest

from src.dp.digital_dp.template import DigitalDP


class TestGeneral(unittest.TestCase):

    def test_digital_dp(self):

        dd = DigitalDP()
        cnt = [0] * 10
        n = 1000000
        for i in range(1, n + 1):
            for w in str(i):
                cnt[int(w)] += 1

        for d in range(10):
            assert dd.count_digit(n, d) == cnt[d]
            assert dd.count_digit_iteration(n, d) == cnt[d]
            dd.count_digit_iteration(10**9, d)
        for d in range(1, 10):
            ans1 = dd.count_num_base(n, d)
            ans2 = sum(str(d) not in str(num) for num in range(1, n + 1))
            assert ans1 == ans2

        for d in range(10):
            ans1 = dd.count_num_dp(n, d)
            ans2 = sum(str(d) not in str(num) for num in range(1, n + 1))
            assert ans1 == ans2

        for d in range(10):
            nums = []
            for i in range(1, n + 1):
                if str(d) not in str(i):
                    nums.append(i)
            for i, num in enumerate(nums):
                assert dd.get_kth_without_d(i + 1, d) == num
        return

    def test_digital_count(self):
        dd = DigitalDP()
        for d in range(100):
            dd.count_digit(10**100, d)
        return

    def test_count_digit_iteration(self):
        dd = DigitalDP()
        for d in range(100):
            dd.count_digit_iteration(10**100, d)
        return

    def test_count_digit_iteration_bfs(self):
        dd = DigitalDP()
        for d in range(100):
            dd.count_digit_bfs(10**100, d)
        return


if __name__ == '__main__':
    unittest.main()
