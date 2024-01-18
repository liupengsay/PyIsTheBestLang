import random
import unittest

from src.strings.kmp.template import KMP


class TestGeneral(unittest.TestCase):

    def test_kmp(self):
        kmp = KMP()
        for _ in range(10):
            for x in range(4):
                lst = [random.randint(0, 10) for _ in range(10 ** x)]
                n = len(lst)
                pi = kmp.prefix_function(lst)
                nxt = kmp.prefix_function_reverse(lst)
                z = kmp.z_function(lst)
                for i in range(1, n):
                    ceil = floor = 0
                    for j in range(1, i + 1):
                        if lst[j:i + 1] == lst[:i - j + 1]:
                            if i - j + 1 > ceil:
                                ceil = i - j + 1
                            if floor == 0 or i - j + 1 < floor:
                                floor = i - j + 1
                    assert pi[i] == ceil
                    assert nxt[i] == floor
                    i1, j1 = 0, i
                    while j1 < n and lst[j1] == lst[i1]:
                        i1 += 1
                        j1 += 1
                    assert z[i] == i1
        return


if __name__ == '__main__':
    unittest.main()
