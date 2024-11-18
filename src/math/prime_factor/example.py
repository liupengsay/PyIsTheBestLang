import unittest

from src.math.prime_factor.template import AllFactorCnt, PrimeFactor, RadFactor


class TestGeneral(unittest.TestCase):


    def test_rad_factor(self):  # 1.891
        n = 2 * 10 ** 5
        rf = RadFactor(n)
        for i in range(n+1):
            assert sorted(rf.get_rad_factor(i)) == sorted(rf.get_rad_factor2(i))
        return


    def test_all_factor(self):  # 1.891
        n = 2 * 10 ** 5
        all_factor = [[], [1]] + [[1, i] for i in range(2, n + 1)]
        for i in range(2, n + 1):
            x = i
            while x * i <= n:
                all_factor[x * i].append(i)
                if i != x:
                    all_factor[x * i].append(x)
                x += 1
        for i in range(n + 1):
            all_factor[i].sort()
        assert [len(ls) for ls in all_factor] == AllFactorCnt(n).all_factor_cnt
        assert all_factor == PrimeFactor(n).all_factor
        return

    def test_prime_factor(self):  # 1.891
        n = 2*10**5
        pf = PrimeFactor(n)
        assert pf.prime_factor_cnt[1:] == [len(ls) for ls in pf.prime_factor[1:]]
        assert pf.prime_factor_mi_cnt[1:] == [sum(x for _, x in ls) for ls in pf.prime_factor[1:]]
        return

if __name__ == '__main__':
    unittest.main()
