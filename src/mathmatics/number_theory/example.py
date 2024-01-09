import random
import unittest

from src.mathmatics.number_theory.template import PrimeSieve, PrimeJudge, NumFactor, RomeToInt, NumBase, \
    EulerPhi
from src.mathmatics.prime_factor.template import PrimeFactor


class TestGeneral(unittest.TestCase):

    def test_prime_sieve(self):
        n = 10 ** 6
        pf = PrimeFactor(n)
        ps = PrimeSieve()
        euler_sieve = ps.euler_sieve(n)
        eratosthenes_sieve = ps.eratosthenes_sieve(n)
        prime_numbers = pf.get_prime_numbers()
        assert euler_sieve == eratosthenes_sieve == prime_numbers
        return

    def test_prime_sieve_euler_sieve(self):  # 699 ms
        n = 10 ** 7
        ps = PrimeSieve()
        ps.euler_sieve(n)
        return

    def test_prime_sieve_eratosthenes_sieve(self):  # 355 ms
        n = 10 ** 7
        ps = PrimeSieve()
        ps.eratosthenes_sieve(n)
        return

    def test_prime_judge(self):
        n = 10 ** 6
        pj = PrimeJudge()
        for num in range(1, n + 1):
            assert pj.is_prime_general(num) == pj.is_prime_speed(num) == pj.is_prime_random(num)
        return

    def test_prime_judge_is_prime_general(self):  # 13 sec 799 ms
        n = 10 ** 7
        pj = PrimeJudge()
        for num in range(1, n + 1):
            pj.is_prime_general(num)
        return

    def test_prime_judge_is_prime_speed(self):  # 4 sec 589 ms
        n = 10 ** 7
        pj = PrimeJudge()
        for num in range(1, n + 1):
            pj.is_prime_speed(num)
        return

    def test_num_factor(self):
        n = 10 ** 5
        pf = PrimeFactor(n)
        nf = NumFactor()
        for num in range(1, n + 1):
            assert pf.all_factor[num] == nf.get_all_factor(num)
            assert pf.prime_factor[num] == nf.get_prime_factor(num)
            cnt = nf.get_prime_with_pollard_rho(num)
            assert pf.prime_factor[num] == [(p, cnt[p]) for p in sorted(cnt)]
            assert pf.all_factor[num] == nf.get_all_with_pollard_rho(num)
        return

    def test_num_factor_get_general(self):  # 1 sec 632 ms
        n = 10 ** 6
        nf = NumFactor()
        for num in range(1, n + 1):
            nf.get_prime_factor(num)
        return

    def test_num_factor_get_with_pollard_rho(self):  # 9 sec 470 ms
        n = 10 ** 6
        nf = NumFactor()
        for num in range(1, n + 1):
            nf.get_prime_with_pollard_rho(num)
        return

    def test_num_factor_get_general_larger(self):  # 1 sec 55 ms
        n = 10 ** 3
        start = 10 ** 12
        nf = NumFactor()
        for num in range(start, n + start):
            nf.get_prime_factor(num)
        return

    def test_num_factor_get_with_pollard_rho_larger(self):  # 201 ms
        n = 10 ** 3
        start = 10 ** 12
        nf = NumFactor()
        for num in range(start, n + start):
            nf.get_prime_with_pollard_rho(num)
        return

    def test_num_factor_get_with_pollard_rho_performance(self):  # 6 sec 769 ms
        n = 10 ** 4
        start = 10 ** 18
        nf = NumFactor()
        for num in range(start, n + start):
            nf.get_prime_with_pollard_rho(num)
        return

    def test_euler_phi(self):
        ep = EulerPhi()
        for num in range(1, 10 ** 6):
            assert ep.euler_phi_general(num) == ep.euler_phi_with_prime_factor(num)
        return

    def test_euler_phi_general(self):  # 5 sec 427 ms
        ep = EulerPhi()
        for num in range(1, 10 ** 6):
            ep.euler_phi_general(num)
        return

    def test_euler_phi_general_larger(self):  # 7 sec 937 ms
        ep = EulerPhi()
        n = 10 ** 3
        start = 10 ** 12
        for num in range(start, n + start):
            ep.euler_phi_general(num)
        return

    def test_euler_phi_with_prime_factor(self):  # 1 sec 738 ms
        ep = EulerPhi()
        for num in range(1, 10 ** 6):
            ep.euler_phi_with_prime_factor(num)
        return

    def test_euler_phi_with_prime_factor_larger(self):  # 233 ms
        ep = EulerPhi()
        n = 10 ** 3
        start = 10 ** 12
        for num in range(start, n + start):
            ep.euler_phi_with_prime_factor(num)
        return

    def test_euler_phi_with_prime_factor_performance(self):  # 6 sec 959 ms
        n = 10 ** 4
        start = 10 ** 18
        ep = EulerPhi()
        for num in range(start, n + start):
            ep.euler_phi_with_prime_factor(num)
        return

    def test_num_base(self):
        nt = NumBase()
        num = random.randint(1, 100)
        assert nt.get_k_bin_of_n(num, 2) == [int(w) for w in bin(num)[2:]]
        assert nt.get_k_bin_of_n(4, -2) == [1, 0, 0]
        return

    def test_rome_to_int(self):
        nt = RomeToInt()

        num = 1000
        for i in range(1, num + 1):
            assert nt.roman_to_int(nt.int_to_roman(i)) == i
        return


if __name__ == '__main__':
    unittest.main()
