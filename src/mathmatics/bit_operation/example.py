import unittest

from src.mathmatics.bit_operation.template import BitOperation


class TestGeneral(unittest.TestCase):

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
            assert bo.sum_xor_2(i) == pre
        return


if __name__ == '__main__':
    unittest.main()
