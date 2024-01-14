import unittest


class TestGeneral(unittest.TestCase):

    def test_prime_factor(self):  # 1.891
        n = 2*10 ** 5
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
        return

    def test_prime_factor3(self):  # 1.891
        n = 10 ** 6
        all_factor1 = [[], [1]] + [[1] for _ in range(2, n + 1)]
        all_factor2 = [[], [1]] + [[i] for i in range(2, n + 1)]
        for i in range(2, n + 1):
            x = i
            while x * i <= n:
                all_factor1[x * i].append(i)
                if i != x:
                    all_factor2[x * i].append(x)
                x += 1
        all_factor = [all_factor1[i]+all_factor2[i][::-1] for i in range(n+1)]
        return

    def test_prime_factor2(self):  # 2.703
        n = 2*10 ** 5
        all_factor = [[1] for _ in range(n + 1)]
        for i in range(2, n + 1):
            x = 1
            while x * i <= n:
                all_factor[x * i].append(i)
                x += 1
        return


if __name__ == '__main__':
    unittest.main()
