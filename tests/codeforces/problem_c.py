import heapq
import random
import sys
from math import inf


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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        # 模板：经典最短路生成树模板题
        n, m, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_ints()
            x -= 1
            y -= 1
            dct[x].append([y, w, ind])
            dct[y].append([x, w, ind])

        for i in range(n):
            dct[i].sort()

        # 先跑一遍最短路
        dis = [inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w, _ in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])

        # 选择字典序较小的边建立最短路树
        edge = [[] for _ in range(n)]
        visit = [0] * n
        for i in range(n):
            for j, w, ind in dct[i]:
                if visit[j]:
                    continue
                if dis[i] + w == dis[j]:
                    edge[i].append([j, ind])
                    edge[j].append([i, ind])
                    visit[j] = 1

        # 最后一遍BFS确定选择的边
        ans = []
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            for y, ind in edge[x]:
                if y != fa:
                    ans.append(ind+1)
                    stack.append([y, x])
        ans = ans[:k]
        ac.st(len(ans))
        ac.lst(ans)
        return


Solution().main()
