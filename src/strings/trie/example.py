import random
import unittest
from collections import Counter

from src.strings.trie.template import BinaryTrie, StringTrie


class TestGeneral(unittest.TestCase):

    def test_binary_trie(self):
        for mi in range(10):
            max_num = 10 ** mi
            num_cnt = 10 ** 4
            bt = BinaryTrie(max_num, num_cnt)
            nums = []
            for i in range(num_cnt):
                x = random.randint(0, 1)
                if x == 0 and nums:
                    j = random.randint(0, len(nums) - 1)
                    assert bt.remove(nums.pop(j))
                else:
                    num = random.randint(0, max_num)
                    nums.append(num)
                    assert bt.add(num)
                dct = Counter(nums)
                for num in dct:
                    assert bt.count(num) == dct[num]
                assert bt.son_and_cnt[0] & bt.mask == len(nums)
                if nums:
                    num = random.randint(0, max_num)
                    lst = [num ^ x for x in nums]
                    lst.sort(reverse=True)
                    assert bt.get_maximum_xor(num) == lst[0]
                    assert bt.get_minimum_xor(num) == lst[-1]
                    res = [bt.get_kth_maximum_xor(num, rk + 1) for rk in range(len(lst))]
                    assert res == lst

                    y = random.randint(0, max_num)
                    assert bt.get_cnt_smaller_xor(num, y) == sum(num ^ x <= y for x in nums)
        return

    def test_string_trie(self):
        for _ in range(10):
            word_cnt = 10 ** 4
            word_length = 10
            st = StringTrie(word_cnt * word_length, word_cnt)
            words = []
            for i in range(word_cnt):
                word = "".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random.randint(1, word_length)))
                words.append(word)
                st.add(word)
            for i in range(word_cnt):
                word = "".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random.randint(1, word_length)))
                res = 0
                for s in words:
                    for j in range(min(len(word), len(s))):
                        if word[j] == s[j]:
                            res += 1
                        else:
                            break
                assert res == st.count(word)
        return


if __name__ == '__main__':
    unittest.main()
