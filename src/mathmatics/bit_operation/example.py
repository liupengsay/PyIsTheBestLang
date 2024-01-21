import random
import unittest

from src.mathmatics.bit_operation.template import BitOperation, MinimumPairXor
from src.utils.fast_io import inf


class TestGeneral(unittest.TestCase):

    def test_minimum_pair(self):
        for x in range(3):
            n = 5 * 10 ** x
            minimum_xor = MinimumPairXor()
            nums = []
            for _ in range(n):
                num = random.randint(0, n)
                minimum_xor.add(num)
                nums.append(num)
                if len(nums) >= 2:
                    c = len(nums)
                    assert [minimum_xor.lst[i] for i in range(c)] == sorted(nums)
                    floor = inf
                    for a in range(c):
                        for b in range(a + 1, c):
                            cur = nums[a] ^ nums[b]
                            if cur < floor:
                                floor = cur
                    assert floor == minimum_xor.query()
        return

    def test_bit_operation(self):
        bo = BitOperation()

        lst = [bo.integer_to_graycode(i) for i in range(11)]
        print(lst)

        assert bo.integer_to_graycode(0) == "0"
        assert bo.integer_to_graycode(22) == "11101"
        assert bo.graycode_to_integer("10110") == 27

        n = 8
        code = bo.get_graycode(n)
        m = len(code)
        for i in range(m):
            assert bo.graycode_to_integer(bin(code[i])[2:]) == i
            assert bo.integer_to_graycode(i) == bin(code[i])[2:]

        pre = 0
        for i in range(100000):
            pre ^= i
            assert bo.sum_xor(i) == pre
        return


if __name__ == '__main__':
    unittest.main()
