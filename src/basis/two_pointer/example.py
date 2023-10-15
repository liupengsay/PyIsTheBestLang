
class TestGeneral(unittest.TestCase):

    def test_two_pointer(self):
        nt = TwoPointer()
        nums = [1, 2, 3, 4, 4, 3, 3, 2, 1, 6, 3]
        assert nt.same_direction(nums) == 4

        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert nt.opposite_direction(nums, 9)
        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert not nt.opposite_direction(nums, 16)
        return

    def test_ops_es(self):

        dct = {max: 0, min: INF, gcd: 0, or_: 0, xor: 0, add: 0, mul: 1, and_: (1 << 32)-1}
        for op in dct:
            for _ in range(1000):
                e = dct[op]
                n = 100
                nums = [random.randint(0, 10**9) for _ in range(n)]
                swa = SlidingWindowAggregation(e, op)
                k = random.randint(1, 50)
                ans = []
                res = []
                for i in range(n):
                    swa.append(nums[i])
                    if i >= k-1:
                        ans.append(swa.query())
                        swa.popleft()
                        lst = nums[i-k+1: i+1]
                        res.append(reduce(op, lst))
                assert len(res) == len(ans)
                assert res == ans



if __name__ == '__main__':
    unittest.main()
