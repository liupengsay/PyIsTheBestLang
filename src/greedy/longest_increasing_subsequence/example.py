
class TestGeneral(unittest.TestCase):

    def test_longest_increasing_subsequence(self):
        lis = LongestIncreasingSubsequence()
        nums = [1, 2, 3, 3, 2, 2, 1]
        assert lis.definitely_increase(nums) == 3
        assert lis.definitely_not_reduce(nums) == 4
        assert lis.definitely_reduce(nums) == 3
        assert lis.definitely_not_increase(nums) == 5

        for _ in range(10):
            nums = [random.randint(0, 100) for _ in range(10)]
            ans = LcsLis().longest_increasing_subsequence_max_sum(nums)
            cur = defaultdict(int)
            n = len(nums)
            for i in range(1, 1 << n):
                lst = [nums[j] for j in range(n) if i & (1 << j)]
                m = len(lst)
                if lst == sorted(lst) and all(lst[j+1] > lst[j] for j in range(m-1)):
                    a, b = cur[m], sum(lst)
                    cur[m] = a if a > b else b
            length = max(cur)
            assert ans == cur[length]
        return


if __name__ == '__main__':
    unittest.main()
