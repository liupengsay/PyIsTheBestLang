import math
import random
import time
import unittest

from src.mathmatics.number_theory.template import NumberTheory


class TestGeneral(unittest.TestCase):

    def test_get_prime_factor(self):
        for i in range(1, 10):
            x = random.randint(i, 10 ** 10)
            t0 = time.time()
            cnt1 = NumberTheory().get_prime_factor(x)
            t1 = time.time()
            cnt2 = NumberTheory().get_prime_factor(x)
            t2 = time.time()
            print(t1 - t0, t2 - t1)
            assert cnt1 == cnt2

    def test_get_prime_factor_pollard(self):
        nt = NumberTheory()
        for i in range(1, 100000):
            res = nt.get_prime_factor(i)
            cnt = nt.get_prime_factors_with_pollard_rho(i)
            num = 1
            for val, c in res:
                num *= val ** c
                if val > 1:
                    assert cnt[val] == c
            assert num == i

        nt = NumberTheory()
        num = 2
        assert nt.get_prime_factor(num) == [[2, 1]]
        num = 1
        assert nt.get_prime_factor(num) == []
        num = 2 * (3 ** 2) * 7 * (11 ** 3)
        assert nt.get_prime_factor(num) == [[2, 1], [3, 2], [7, 1], [11, 3]]
        return

    def test_euler_phi(self):
        nt = NumberTheory()
        assert nt.euler_phi(10 ** 11 + 131) == 66666666752
        return

    def test_euler_shai(self):
        nt = NumberTheory()
        label = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        pred = nt.euler_flag_prime(30)
        assert label == pred
        assert len(nt.euler_flag_prime(10 ** 6)) == 78498
        return

    def test_eratosthenes_shai(self):
        nt = NumberTheory()
        assert len(nt.sieve_of_eratosthenes(10 ** 6)) == 78498
        return

    def test_factorial_zero_count(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        s = str(math.factorial(num))
        cnt = 0
        for w in s[::-1]:
            if w == "0":
                cnt += 1
            else:
                break
        assert nt.factorial_zero_count(num) == cnt
        return

    def test_get_k_bin_of_n(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        assert nt.get_k_bin_of_n(num, 2) == [int(w) for w in bin(num)[2:]]

        assert nt.get_k_bin_of_n(4, -2) == [1, 0, 0]
        return

    def test_rational_number_to_fraction(self):
        nt = NumberTheory()
        assert nt.rational_number_to_fraction("33") == [1, 3]
        return

    def test_is_prime(self):
        nt = NumberTheory()
        assert not nt.is_prime(1)
        assert nt.is_prime(5)
        assert not nt.is_prime(51)
        for _ in range(10):
            i = random.randint(1, 10 ** 3)
            assert nt.is_prime(i) == nt.is_prime4(i) == nt.is_prime5(i)

        for _ in range(1):
            x = random.randint(10 ** 8, 10 ** 9)
            y = x + 10 ** 6
            for num in range(x, y + 1):
                nt.is_prime4(x)
        return

    def test_gcd_lcm(self):
        nt = NumberTheory()
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        assert nt.gcd(a, b) == math.gcd(a, b)
        assert nt.lcm(a, b) == math.lcm(a, b)
        return

    def test_get_factor(self):
        nt = NumberTheory()

        num = 1000
        ans = nt.get_factor_upper(num)
        for i in range(1, num + 1):
            assert ans[i] == nt.get_all_factor(i)[1:-1]
        return

    def test_roma_int(self):
        nt = NumberTheory()

        num = 1000
        for i in range(1, num + 1):
            assert nt.roman_to_int(nt.int_to_roman(i)) == i
        return


if __name__ == '__main__':
    unittest.main()
