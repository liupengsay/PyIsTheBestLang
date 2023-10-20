import random
import unittest


class TestGeneral(unittest.TestCase):

    def test_string_hash(self):
        n = 1000
        st = "".join([chr(random.randint(0, 25) + ord("a")) for _ in range(n)])

        # 生成哈希种子
        p1, p2 = random.randint(26, 100), random.randint(26, 100)
        mod1, mod2 = random.randint(
            10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        # 计算目标串的哈希状态
        target = "".join([chr(random.randint(0, 25) + ord("a"))
                          for _ in range(10)])
        h1 = h2 = 0
        for w in target:
            h1 = h1 * p1 + (ord(w) - ord("a"))
            h1 %= mod1
            h2 = h2 * p2 + (ord(w) - ord("a"))
            h2 %= mod1

        # 滑动窗口计算哈希状态
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


if __name__ == '__main__':
    unittest.main()
