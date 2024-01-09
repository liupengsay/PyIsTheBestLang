from src.utils.fast_io import inf


class TrieKeyWordSearchInText:
    def __init__(self):
        self.dct = dict()
        return

    def add_key_word(self, word, i):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["isEnd"] = i

    def search_text(self, text):
        cur = self.dct
        res = []
        for w in text:
            if w in cur:
                cur = cur[w]
                if "isEnd" in cur:
                    res.append(cur["isEnd"])
            else:
                break
        return res


class BinaryTrie:
    def __init__(self, max_num, num_cnt):
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

    def add(self, num: int) -> bool:
        cur = 0
        self.son_and_cnt[cur] += 1
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            if not self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit:
                self.son_and_cnt[(cur << 1) | bit] |= (self.ind << self.cnt_bit)
                self.ind += 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            self.son_and_cnt[cur] += 1
        return True

    def remove(self, num: int) -> bool:
        if self.son_and_cnt[0] & self.mask == 0:
            return False
        cur = 0
        self.son_and_cnt[0] -= 1
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if cur == 0 or self.son_and_cnt[cur] & self.mask == 0:
                return False
            self.son_and_cnt[cur] -= 1
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


class StringTrie:
    def __init__(self, most_word, word_cnt):
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

# class BinaryTrie:
#     def __init__(self, max_bit: int = 30):
#         self.inf = 1 << 63
#         self.to = [[-1], [-1]]
#         self.cnt = [0]
#         self.max_bit = max_bit
# 
#     def add(self, num: int) -> None:
#         cur = 0
#         self.cnt[cur] += 1
#         for k in range(self.max_bit, -1, -1):
#             bit = (num >> k) & 1
#             if self.to[bit][cur] == -1:
#                 self.to[bit][cur] = len(self.cnt)
#                 self.to[0].append(-1)
#                 self.to[1].append(-1)
#                 self.cnt.append(0)
#             cur = self.to[bit][cur]
#             self.cnt[cur] += 1
#         return
# 
#     def remove(self, num: int) -> bool:
#         if self.cnt[0] == 0:
#             return False
#         cur = 0
#         rm = [0]
#         for k in range(self.max_bit, -1, -1):
#             bit = (num >> k) & 1
#             cur = self.to[bit][cur]
#             if cur == -1 or self.cnt[cur] == 0:
#                 return False
#             rm.append(cur)
#         for cur in rm:
#             self.cnt[cur] -= 1
#         return True
# 
#     def count(self, num: int):
#         cur = 0
#         for k in range(self.max_bit, -1, -1):
#             bit = (num >> k) & 1
#             cur = self.to[bit][cur]
#             if cur == -1 or self.cnt[cur] == 0:
#                 return 0
#         return self.cnt[cur]
# 
#     # Get max result for constant x ^ element in array
#     def max_xor(self, x: int) -> int:
#         if self.cnt[0] == 0:
#             return -self.inf
#         res = cur = 0
#         for k in range(self.max_bit, -1, -1):
#             bit = (x >> k) & 1
#             nxt = self.to[bit ^ 1][cur]
#             if nxt == -1 or self.cnt[nxt] == 0:
#                 cur = self.to[bit][cur]
#             else:
#                 res |= 1 << k
#                 cur = nxt
# 
#         return res
# 
#     # Get min result for constant x ^ element in array
#     def min_xor(self, x: int) -> int:
#         if self.cnt[0] == 0:
#             return self.inf
#         res = cur = 0
#         for k in range(self.max_bit, -1, -1):
#             bit = (x >> k) & 1
#             nxt = self.to[bit][cur]
#             if nxt == -1 or self.cnt[nxt] == 0:
#                 res |= 1 << k
#                 cur = self.to[bit ^ 1][cur]
#             else:
#                 cur = nxt
#         return res
