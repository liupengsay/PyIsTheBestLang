import random
import unittest

from src.strings.manacher_palindrome.template import ManacherPlindrome


class TestGeneral(unittest.TestCase):

    def test_manacher_palindrome_count_start_end(self):
        mp = ManacherPlindrome()
        for x in range(4):
            n = 10 ** x
            for _ in range(10):
                nums = [random.randint(0, x) for _ in range(n)]
                start = [0] * n
                end = [0] * n
                cnt = [0] * (n + 1)
                for i in range(n):
                    for j in range(i, n):
                        if nums[i:j + 1] == nums[i:j + 1][::-1]:
                            start[i] += 1
                            end[j] += 1
                            cnt[j - i + 1] += 1
                assert start, end == mp.palindrome_count_start_end("".join(chr(x + ord("a")) for x in nums))
                assert cnt == mp.palindrome_length_count("".join(chr(x + ord("a")) for x in nums))
                assert sum(cnt) == mp.palindrome_count("".join(chr(x + ord("a")) for x in nums))

        return


if __name__ == '__main__':
    unittest.main()
