import random
import unittest

from src.strings.string_hash.template import PointSetRangeHashReverse, RangeChangeRangeHashReverse


class TestGeneral(unittest.TestCase):

    def test_string_hash(self):
        n = 1000
        st = "".join([chr(random.randint(0, 25) + ord("a")) for _ in range(n)])

        # 生成hash种子
        p1, p2 = random.randint(26, 100), random.randint(26, 100)
        mod1, mod2 = random.randint(
            10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        # 目标串的hash状态
        target = "".join([chr(random.randint(0, 25) + ord("a"))
                          for _ in range(10)])
        h1 = h2 = 0
        for w in target:
            h1 = h1 * p1 + (ord(w) - ord("a"))
            h1 %= mod1
            h2 = h2 * p2 + (ord(w) - ord("a"))
            h2 %= mod1

        # sliding_windowhash状态
        m = len(target)
        pow1 = pow(p1, m - 1, mod1)
        pow2 = pow(p2, m - 1, mod2)
        s1 = s2 = 0
        cnt = 0
        n = len(st)
        for i in range(n):
            w = st[i]
            s1 = s1 * p1 + (ord(w) - ord("a"))
            s1 %= mod1
            s2 = s2 * p2 + (ord(w) - ord("a"))
            s2 %= mod1
            if i >= m - 1:
                if (s1, s2) == (h1, h2):
                    cnt += 1
                s1 = s1 - (ord(st[i - m + 1]) - ord("a")) * pow1
                s1 %= mod1
                s2 = s2 - (ord(st[i - m + 1]) - ord("a")) * pow2
                s2 %= mod1
            if st[i:i + m] == target:
                cnt -= 1
        assert cnt == 0
        return

    def test_point_set_range_hash_reverse(self):

        n = 10 ** 4
        nums = [0] * n
        tree = PointSetRangeHashReverse(n)
        for _ in range(1000):
            i = random.randint(0, n - 1)
            num = random.randint(0, n - 1)
            nums[i] = num
            tree.point_set(i, i, num)
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            res = 0
            for j in range(ll, rr + 1):
                res = (res * tree.p + nums[j]) % tree.mod
            assert res == tree.range_hash(ll, rr)
            res = 0
            for j in range(rr, ll - 1, -1):
                res = (res * tree.p + nums[j]) % tree.mod
            assert res == tree.range_hash_reverse(ll, rr)
        assert nums == tree.get()
        return

    def test_range_change_range_hash_reverse(self):

        n = 10 ** 4
        nums = [0] * n
        tree = RangeChangeRangeHashReverse(n)
        for _ in range(1000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(0, n - 1)
            for i in range(ll, rr + 1):
                nums[i] = num
            tree.range_change(ll, rr, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            res = 0
            for j in range(ll, rr + 1):
                res = (res * tree.p + nums[j]) % tree.mod
            assert res == tree.range_hash(ll, rr)
            res = 0
            for j in range(rr, ll - 1, -1):
                res = (res * tree.p + nums[j]) % tree.mod
            assert res == tree.range_hash_reverse(ll, rr)
        assert nums == tree.get()
        return


if __name__ == '__main__':
    unittest.main()
