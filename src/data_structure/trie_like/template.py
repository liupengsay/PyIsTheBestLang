from src.utils.fast_io import inf


class BinaryTrieXor:
    def __init__(self, max_num, num_cnt):  # bitwise xor
        if max_num <= 0:
            max_num = 1
        if num_cnt <= 0:
            num_cnt = 1
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


class BinaryTrieXorLimited:
    def __init__(self, max_num, num_cnt):  # bitwise xor
        if max_num <= 0:
            max_num = 1
        if num_cnt <= 0:
            num_cnt = 1
        binary_state = 2
        self.max_bit = max_num.bit_length() - 1
        self.cnt_bit = num_cnt.bit_length()
        self.node_cnt = (self.max_bit + 1) * num_cnt * binary_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.floor = [inf] * (self.node_cnt + 1)
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
            if num < self.floor[cur]:
                self.floor[cur] = num
            self.son_and_cnt[cur] += c
        return True

    def get_maximum_xor_limited(self, x: int, m) -> int:
        """get maximum result for constant x ^ element in array and element <= m"""
        if self.son_and_cnt[0] & self.mask == 0:
            return -1
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0 or self.floor[nxt] > m:
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if cur == 0 or self.son_and_cnt[cur] & self.mask == 0 or self.floor[cur] > m:
                    return -1
            else:
                res |= 1 << k
                cur = nxt
        return res


class StringTrieSearch:
    def __init__(self, most_word, word_cnt, string_state=26):  # search index
        # assert most_word >= 1
        # assert word_cnt >= 1
        self.string_state = string_state
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
        # assert 1 <= ind <= word_cnt
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def search(self, word):
        res = []
        cur = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            if self.son_and_ind[cur] & self.mask:
                res.append(self.son_and_ind[cur] & self.mask)
        return res

    def add_ind(self, word, ind):
        # assert 1 <= ind <= word_cnt
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if not self.son_and_ind[cur] & self.mask:
                self.son_and_ind[cur] |= ind
        return

    def search_ind(self, word):
        res = cur = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            if self.son_and_ind[cur] & self.mask:
                res = self.son_and_ind[cur] & self.mask
        return res

    def search_length(self, word):
        cur = res = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            res += 1
        return res

    def add_cnt(self, word, ind):
        cur = res = 0
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if self.son_and_ind[cur] & self.mask:
                res += 1
        self.son_and_ind[cur] |= ind
        res += 1
        return res

    def add_exist(self, word, ind):
        cur = 0
        res = False
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                res = True
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if self.son_and_ind[cur] & self.mask:
                res += 1
        self.son_and_ind[cur] |= ind
        return res

    def add_bin(self, word, ind):
        cur = 0
        for w in word:
            bit = int(w)
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def add_int(self, word, ind):
        cur = 0
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def search_for_one_difference(self, word):
        n = len(word)
        stack = [(0, 0, 0)]
        while stack:
            cur, i, c = stack.pop()
            if i == n:
                return True
            bit = word[i]
            if not c:
                for bit2 in range(26):
                    if bit2 != bit:
                        nex = self.son_and_ind[bit2 + self.string_state * cur] >> self.cnt_bit
                        if nex:
                            stack.append((nex, i + 1, c + 1))
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if cur:
                stack.append((cur, i + 1, c))
        return False


class StringTriePrefix:
    def __init__(self, most_word, word_cnt, string_state=26):  # prefix count
        # assert most_word >= 1
        # assert word_cnt >= 1
        self.string_state = string_state
        self.cnt_bit = word_cnt.bit_length()
        self.node_cnt = most_word * self.string_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.ind = 0
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 0

    def add(self, word, val=1):
        cur = 0  # word: List[int]
        self.son_and_cnt[cur] += val
        for bit in word:
            if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
            self.son_and_cnt[cur] += val
        return

    def count(self, word):
        res = cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur or self.son_and_cnt[cur] & self.mask == 0:
                break
            res += self.son_and_cnt[cur] & self.mask
        return res

    def count_end(self, word):
        cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur or self.son_and_cnt[cur] & self.mask == 0:
                return 0
        return self.son_and_cnt[cur] & self.mask

    def add_end(self, word, val=1):
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_cnt[cur] += val
        return

    def count_pre_end(self, word):
        res = cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            res += self.son_and_cnt[cur] & self.mask
        return res
