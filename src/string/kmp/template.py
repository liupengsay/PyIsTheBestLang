class KMP:
    def __init__(self):
        return

    @classmethod
    def prefix_function(cls, s):
        """calculate the longest common true prefix and true suffix for s [:i+1] and s [:i+1]"""
        n = len(s)  # fail tree
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:  # all pi[i] pi[pi[i]] ... are border
                j += 1  # all i+1-pi[i] pi[i]+1-pi[pi[i]] ... are circular_section
            pi[i] = j  # pi[i] <= i also known as next
        # pi[0] = 0
        return pi  # longest common true prefix_suffix / i+1-nex[i] is shortest circular_section

    @staticmethod
    def z_function(s):
        """calculate the longest common prefix between s[i:] and s"""
        n = len(s)
        z = [0] * n
        left, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - left] < r - i + 1:
                z[i] = z[i - left]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                left = i
                r = i + z[i] - 1
        # z[0] = 0
        return z

    def prefix_function_reverse(self, s):
        n = len(s)
        nxt = [0] + self.prefix_function(s)
        nxt[1] = 0
        for i in range(2, n + 1):
            j = i
            while nxt[j]:
                j = nxt[j]
            if nxt[i]:
                nxt[i] = j
        return nxt[1:]  # shortest common true prefix_suffix / i+1-nex[i] is longest circular_section

    def find(self, s1, s2):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_lst(self, s1, s2, tag=-1):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + [tag] + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_longest_palindrome(self, s, pos="prefix") -> int:
        """calculate the longest prefix and longest suffix palindrome substring"""
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]

    @staticmethod
    def kmp_automaton(s, m=26):
        n = len(s)
        nxt = [0] * m * (n + 1)
        j = 0
        for i in range(1, n + 1):
            j = nxt[j * m + s[i - 1]]
            nxt[(i - 1) * m + s[i - 1]] = i
            for k in range(m):
                nxt[i * m + k] = nxt[j * m + k]
        return nxt

    @classmethod
    def merge_b_from_a(cls, a, b):
        c = b + "#" + a
        f = cls.prefix_function(c)
        m = len(b)
        if max(f[m:]) == m:
            return a
        x = f[-1]
        return a + b[x:]


class InfiniteStream:
    def next(self) -> int:
        pass
