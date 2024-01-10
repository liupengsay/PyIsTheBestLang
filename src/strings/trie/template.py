from src.utils.fast_io import inf


class BinaryTrieXor:
    def __init__(self, max_num, num_cnt):  # bitwise xor
        assert max_num >= 1
        assert num_cnt >= 1
        binary_state = 2
        self.max_bit = max_num.bit_length() - 1
        self.cnt_bit = num_cnt.bit_length()
        self.node_cnt = (self.max_bit + 1) * num_cnt * binary_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.ind = 1
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 1

    def add(self, num: int, c=1) -> bool:
        cur = 0
        self.son_and_cnt[cur] += c
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            if not self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit:
                self.son_and_cnt[(cur << 1) | bit] |= (self.ind << self.cnt_bit)
                self.ind += 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            self.son_and_cnt[cur] += c
        return True

    def remove(self, num: int, c=1) -> bool:
        if self.son_and_cnt[0] & self.mask < c:
            return False
        cur = 0
        self.son_and_cnt[0] -= c
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if cur == 0 or self.son_and_cnt[cur] & self.mask < c:
                return False
            self.son_and_cnt[cur] -= c
        return True

    def count(self, num: int):
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if cur == 0 or self.son_and_cnt[cur] & self.mask == 0:
                return 0
        return self.son_and_cnt[cur] & self.mask

    def get_maximum_xor(self, x: int) -> int:
        """get maximum result for constant x ^ element in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return -inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            else:
                res |= 1 << k
                cur = nxt
        return res

    def get_minimum_xor(self, x: int) -> int:
        """get minimum result for constant x ^ element in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                res |= 1 << k
                cur = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            else:
                cur = nxt
        return res

    def get_kth_maximum_xor(self, x: int, rk) -> int:
        """get kth maximum result for constant x ^ element in array"""
        assert rk >= 1
        if self.son_and_cnt[0] & self.mask < rk:
            return -inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask < rk:
                if nxt:
                    rk -= self.son_and_cnt[nxt] & self.mask
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            else:
                res |= 1 << k
                cur = nxt
        return res

    def get_cnt_smaller_xor(self, x: int, y: int) -> int:
        """get cnt result for constant x ^ element <= y in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return 0
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            if not (y >> k) & 1:
                nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                    return res
                cur = nxt
            else:
                nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if nxt:
                    res += self.son_and_cnt[nxt] & self.mask
                nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
                if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                    return res
                cur = nxt
        res += self.son_and_cnt[cur] & self.mask
        return res


class StringTrieSearch:
    def __init__(self, most_word, word_cnt):  # search index
        assert most_word >= 1
        assert word_cnt >= 1
        self.string_state = 26
        self.cnt_bit = word_cnt.bit_length()
        self.node_cnt = most_word * self.string_state
        self.son_and_ind = [0] * (self.node_cnt + 1)
        self.ind = 0
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_ind[i] = 0
        self.ind = 0

    def add(self, word, ind):
        cur = 0
        for w in word:
            bit = ord(w) - ord("a")
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def search(self, word):
        res = []
        cur = 0
        for w in word:
            bit = ord(w) - ord("a")
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            if self.son_and_ind[cur] & self.mask:
                res.append(self.son_and_ind[cur] & self.mask)
        return res


class StringTriePrefix:
    def __init__(self, most_word, word_cnt):  # prefix count
        assert most_word >= 1
        assert word_cnt >= 1
        self.string_state = 26
        self.cnt_bit = word_cnt.bit_length()
        self.node_cnt = most_word * self.string_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.ind = 0
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 0

    def add(self, word):
        cur = 0
        self.son_and_cnt[cur] += 1
        for w in word:
            bit = ord(w) - ord("a")
            if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
            self.son_and_cnt[cur] += 1
        return

    def count(self, word):
        res = cur = 0
        for w in word:
            bit = ord(w) - ord("a")
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur or self.son_and_cnt[cur] & self.mask == 0:
                break
            res += self.son_and_cnt[cur] & self.mask
        return res
