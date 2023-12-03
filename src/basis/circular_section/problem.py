"""
算法：循环节
功能：通过模拟找出循环节进行状态计算
题目：

====================================LeetCode====================================
957（https://leetcode.com/problems/prison-cells-after-n-days/）循环节计算
418（https://leetcode.com/problems/sentence-screen-fitting/）循环节计算
466（https://leetcode.com/problems/count-the-repetitions/）循环节计算
1806（https://leetcode.com/problems/minimum-number-of-operations-to-reinitialize-a-permutation/description/）根据有限状态判断循环节的大小

=====================================LuoGu======================================
1965（https://www.luogu.com.cn/problem/P1965）循环节计算
1532（https://www.luogu.com.cn/problem/P1532）循环节计算
2203（https://www.luogu.com.cn/problem/P2203）循环节计算
5550（https://www.luogu.com.cn/problem/P5550）循环节计算也可以使用矩阵快速幂递推
7318（https://www.luogu.com.cn/problem/P7318）二维元素，再增加虚拟开始状态，进行循环节计算
7681（https://www.luogu.com.cn/problem/P7681）带前缀和的循环节，注意定义循环状态
1468（https://www.luogu.com.cn/problem/P1468）状态压缩求循环节
6148（https://www.luogu.com.cn/problem/P6148）经典计算循环节后模拟

===================================CodeForces===================================
C. Yet Another Counting Problem（https://codeforces.com/problemset/problem/1342/C）循环节计数
B. Jellyfish and Game（https://codeforces.com/contest/1875/problem/B）circle section find with hash and list

参考：OI WiKi（xx）
"""
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_957(cells: List[int], n: int) -> List[int]:
        # 模板：N 天后的牢房经典循环节
        def compute_loop(i, j, n):
            # 此时只需计算k即可，即最后一次的状态
            if j == n:
                return n
            k = i + (n - i) % (j - i)
            return k

        m = len(cells)
        dct = dict()
        state = []
        day = 0
        # 进行模拟状态
        while day < n:
            busy = set([i for i in range(1, m - 1)
                        if cells[i - 1] == cells[i + 1]])
            cells = [1 if i in busy else 0 for i in range(m)]
            day += 1
            state.append(cells[:])
            if tuple(cells) in dct:
                break
            dct[tuple(cells)] = day

        # 计算循环节信息
        i = dct[tuple(cells)]
        j = day
        k = compute_loop(i, j, n)
        return state[k - 1]

    @staticmethod
    def lc_1806(n: int) -> int:
        # 模板：根据有限状态判断循环节的大小
        ans = 0
        visit = [0] * n
        for i in range(n):
            if visit[i]:
                continue
            cur = 0
            x = i
            while not visit[x]:
                visit[x] = 1
                cur += 1
                if x % 2 == 0:
                    x //= 2
                else:
                    x = n // 2 + (x - 1) // 2
            if cur > ans:
                ans = cur
        return ans

    @staticmethod
    def lg_p1468(ac=FastIO()):
        # 模板：状态压缩求循环节
        n = ac.read_int()
        op1 = (1 << n) - 1
        op2 = sum(1 << i for i in range(0, n, 2))
        op3 = sum(1 << i for i in range(1, n, 2))
        op4 = sum(1 << i for i in range(0, n, 3))
        stack = [[(1 << n) - 1]]
        # 进行所有的操作模拟与循环节计算
        ans = []
        while stack:
            path = stack.pop()
            for op in [op1, op2, op3, op4]:
                if path[-1] ^ op in path:
                    # 遇到循环
                    ans.append(path[:])
                else:
                    stack.append(path + [path[-1] ^ op])

        # 匹配开关状态一致的路径
        c = ac.read_int()
        light = sum(1 << (i - 1) for i in ac.read_list_ints()[:-1])
        down = sum(1 << (i - 1) for i in ac.read_list_ints()[:-1])
        res = set()
        for p in ans:
            m = len(p)
            state = p[c % m]
            if state & light == light and state & down == 0:
                r = bin(state)[2:]
                res.add("0" * (n - len(r)) + r)
        if not res:
            ac.st("IMPOSSIBLE")
            return
        res = sorted([r[::-1] for r in res])
        for r in res:
            ac.st(r)
        return

    @staticmethod
    def lg_p6148(ac=FastIO()):
        # 模板：经典计算循环节后模拟
        n, m, k = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(m)]
        nex = [-1] * n
        for i in range(n):
            x = i
            for a, b in nums:
                if a <= x <= b:
                    x = a + b - x
            nex[i] = x
        # 找出循环节
        ans = [0] * n
        for i in range(n):
            if ans[i]:
                continue
            lst = [i]
            while nex[lst[-1]] != lst[0]:
                lst.append(nex[lst[-1]])
            m = len(lst)
            for j in range(m):
                # 进行相应的移动
                ans[lst[(j + k) % m]] = lst[j] + 1
        for a in ans:
            ac.st(a)
        return