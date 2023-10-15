




class TestGeneral(unittest.TestCase):

    def test_trie_count(self):
        tc = TrieCount()
        words = ["happy", "hello", "leetcode", "let"]
        for word in words:
            tc.update(word)
        assert tc.query("h") == 2
        assert tc.query("le") == 2
        assert tc.query("lt") == 0
        return

    def test_trie_xor_kth_max(self):
        nums = [random.randint(0, 10**9) for _ in range(1000)]
        n = len(nums)
        trie = TrieZeroOneXorMaxKth(len(bin(max(nums))))
        for num in nums:
            trie.add(num)
        for _ in range(10):
            x = random.randint(0, n - 1)
            lst = [nums[x] ^ nums[i] for i in range(n)]
            lst.sort(reverse=True)
            for i in range(n):
                assert lst[i] == trie.query_xor_kth_max(nums[x], i + 1)

        n = 1000
        nums = [0] + [random.randint(0, 100000) for _ in range(n)]
        for i in range(1, n+1):
            nums[i] ^= nums[i-1]
        trie = TrieZeroOneXorMaxKth(len(bin(max(nums))))
        for i, num in enumerate(nums):
            trie.add(num)

        for i in range(n + 1):
            lst = [nums[i]^nums[j] for j in range(n+1)]
            lst.sort(reverse=True)
            for j in range(n+1):
                assert lst[j] == trie.query_xor_kth_max(nums[i], j+1)
        return

if __name__ == '__main__':
    unittest.main()
