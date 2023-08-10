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


def dijkstra_src_to_dst_path(dct, src: int, dst: int):
    # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
    n = len(dct)
    dis = [inf] * n
    stack = [[0, src]]
    dis[src] = 0
    father = [-1] * n  # 记录最短路的上一跳
    while stack:
        d, i = heapq.heappop(stack)
        if dis[i] < d:
            continue
        if i == dst:
            break
        for j, w in dct[i]:
            dj = w + d
            if dj < dis[j]:
                dis[j] = dj
                father[j] = i
                heapq.heappush(stack, [dj, j])
    if dis[dst] == inf:
        return [], inf
    # 向上回溯路径
    path = []
    i = dst
    while i != -1:
        path.append(i)
        i = father[i]
    return path, dis[dst]


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            dct[j].append([i, w])
        path, ans = dijkstra_src_to_dst_path(dct, 0, n-1)
        if ans == inf:
            ac.st(-1)
        else:
            path.reverse()
            ac.lst([x+1 for x in path])
        return


Solution().main()
