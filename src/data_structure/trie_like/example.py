import random
import unittest
from collections import Counter

from src.data_structure.trie_like.problem import Solution
from src.data_structure.trie_like.template import BinaryTrieXor, StringTriePrefix


class TestGeneral(unittest.TestCase):

    def test_binary_trie(self):
        random.seed(2024)
        for mi in range(10):
            max_num = 10 ** mi
            num_cnt = 5 * 10 ** 3
            trie = BinaryTrieXor(max_num, num_cnt)
            nums = []
            for i in range(num_cnt):
                x = random.randint(0, 1)
                if x == 0 and nums:
                    j = random.randint(0, len(nums) - 1)
                    c = min(random.randint(1, nums.count(nums[j])), 3)
                    num = nums[j]
                    for _ in range(c):
                        nums.remove(num)
                    assert trie.remove(num, c)
                else:
                    num = random.randint(0, max_num)
                    c = random.randint(1, 3)
                    for _ in range(c):
                        nums.append(num)
                    assert trie.add(num, c)
                dct = Counter(nums)
                for num in dct:
                    assert trie.count(num) == dct[num]
                assert trie.son_and_cnt[0] & trie.mask == len(nums)
                if nums:
                    num = random.randint(0, max_num)
                    lst = [num ^ x for x in nums]
                    lst.sort(reverse=True)
                    assert trie.get_maximum_xor(num) == lst[0]
                    assert trie.get_minimum_xor(num) == lst[-1]
                    res = [trie.get_kth_maximum_xor(num, rk + 1) for rk in range(len(lst))]
                    assert res == lst

                    y = random.randint(0, max_num)
                    assert trie.get_cnt_smaller_xor(num, y) == sum(num ^ x <= y for x in nums)
        return

    def test_string_trie(self):
        for _ in range(10):
            word_cnt = 10 ** 4
            word_length = 10
            trie = StringTriePrefix(word_cnt * word_length, word_cnt)
            words = []
            for i in range(word_cnt):
                word = "".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random.randint(1, word_length)))
                words.append(word)
                trie.add(word)
            for i in range(word_cnt):
                word = "".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random.randint(1, word_length)))
                res = 0
                for s in words:
                    for j in range(min(len(word), len(s))):
                        if word[j] == s[j]:
                            res += 1
                        else:
                            break
                assert res == trie.count(word)
        return

    def test_solution_lc_421_1(self):  # 411 ms
        random.seed(2024)
        nums = [random.randint(0, (1 << 31) - 1) for _ in range(2 * 10 ** 5)]
        Solution().lc_421_1(nums)
        nums = list(range(2 * 10 ** 5))
        Solution().lc_421_1(nums)
        return

    def test_solution_lc_421_2(self):  # 247 ms
        random.seed(2024)
        nums = [random.randint(0, (1 << 31) - 1) for _ in range(2 * 10 ** 5)]
        Solution().lc_421_2(nums)
        nums = list(range(2 * 10 ** 5))
        Solution().lc_421_2(nums)
        return

    def test_solution_lc_421(self):
        random.seed(2024)
        nums = [random.randint(0, (1 << 31) - 1) for _ in range(2 * 10 ** 5)]
        assert Solution().lc_421_1(nums) == Solution().lc_421_2(nums)
        return


if __name__ == '__main__':
    unittest.main()
