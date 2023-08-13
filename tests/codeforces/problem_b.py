import random
import sys
from collections import defaultdict


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_ints():
        return map(int, sys.stdin.readline().strip().split())

    @staticmethod
    def read_floats():
        return map(float, sys.stdin.readline().strip().split())

    @staticmethod
    def read_ints_minus_one():
        return map(lambda x: int(x) - 1, sys.stdin.readline().strip().split())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def ask(self, lst):
        # CF交互题输出询问并读取结果
        self.lst(lst)
        sys.stdout.flush()
        res = self.read_int()
        # 记得任何一个输出之后都要 sys.stdout.flush() 刷新
        return res

    def out_put(self, lst):
        # CF交互题输出最终答案
        self.lst(lst)
        sys.stdout.flush()
        return

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    @staticmethod
    def get_random_seed():
        # 随机种子避免哈希冲突
        return random.randint(0, 10**9+7)

class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size

class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n, m = ac.read_ints()
        dct =[[] for _ in range(n)]
        uf = UnionFind(n)
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            uf.union(i, j)
        if uf.part != 1:
            ac.st("NO")
        if m == n:
            ac.st("YES")
        else:
            ac.st("NO")
        return


Solution().main()
