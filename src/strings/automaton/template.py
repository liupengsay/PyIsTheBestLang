from collections import deque



class Node:
    __slots__ = 'son', 'fail', 'last', 'len', 'val'

    def __init__(self):
        self.son = {}
        self.fail = self.last = None
        self.len = 0
        self.val = math.inf


class AhoCorasick:
    def __init__(self):
        self.root = Node()

    def insert(self, word, cost):
        x = self.root
        for c in word:
            if c not in x.son:
                x.son[c] = Node()
            x = x.son[c]
        x.len = len(word)
        x.val = min(x.val, cost)

    def set_fail(self):
        q = deque()
        for x in self.root.son.values():
            x.fail = x.last = self.root
            q.append(x)
        while q:
            x = q.popleft()
            for c, son in x.son.items():
                p = x.fail
                while p and c not in p.son:
                    p = p.fail
                son.fail = p.son[c] if p else self.root
                son.last = son.fail if son.fail.len else son.fail.last
                q.append(son)

    def search(self, target):
        pos = [[] for _ in range(len(target))]
        x = self.root
        for i, c in enumerate(target):
            while x and c not in x.son:
                x = x.fail
            x = x.son[c] if x else self.root
            cur = x
            while cur:
                if cur.len:
                    pos[i - cur.len + 1].append(cur.val)
                cur = cur.last
        return pos


class AcAutomaton:
    def __init__(self, p):
        self.m = sum(len(t) for t in p)
        self.n = len(p)
        self.p = p
        self.tr = [[0] * 26 for _ in range(self.m + 1)]
        self.end = [0] * (self.m + 1)
        self.fail = [0] * (self.m + 1)
        self.cnt = 0
        for i, t in enumerate(self.p):
            self.insert(i + 1, t)
        self.set_fail()
        return

    def insert(self, i: int, word: str):
        x = 0
        for c in word:
            c = ord(c) - ord('a')
            if self.tr[x][c] == 0:
                self.cnt += 1
                self.tr[x][c] = self.cnt
            x = self.tr[x][c]
        self.end[i] = x

    def search(self, s):
        freq = [0] * (self.cnt + 1)
        x = 0
        for c in s:
            x = self.tr[x][ord(c) - ord('a')]
            freq[x] += 1

        rg = [[] for _ in range(self.cnt + 1)]
        for i in range(self.cnt + 1):
            rg[self.fail[i]].append(i)

        vis = [False] * (self.cnt + 1)
        st = [0]
        while st:
            x = st[-1]
            if not vis[x]:
                vis[x] = True
                for y in rg[x]:
                    st.append(y)
            else:
                st.pop()
                for y in rg[x]:
                    freq[x] += freq[y]

        res = [freq[self.end[i]] for i in range(1, self.n + 1)]
        return res

    def set_fail(self):
        q = deque([self.tr[0][i] for i in range(26) if self.tr[0][i]])
        while q:
            x = q.popleft()
            for i in range(26):
                if self.tr[x][i] == 0:
                    self.tr[x][i] = self.tr[self.fail[x]][i]
                else:
                    self.fail[self.tr[x][i]] = self.tr[self.fail[x]][i]
                    q.append(self.tr[x][i])
        return
