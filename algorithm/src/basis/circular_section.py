import unittest
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：循环节
功能：通过模拟找出循环节进行状态计算
题目：

===================================力扣===================================
957. N 天后的牢房（https://leetcode.cn/problems/prison-cells-after-n-days/）循环节计算
418. 屏幕可显示句子的数量（https://leetcode.cn/problems/sentence-screen-fitting/）循环节计算
466. 统计重复个数（https://leetcode.cn/problems/count-the-repetitions/）循环节计算

===================================洛谷===================================
P1965 [NOIP2013 提高组] 转圈游戏（https://www.luogu.com.cn/problem/P1965）循环节计算
P1532 卡布列克圆舞曲（https://www.luogu.com.cn/problem/P1532）循环节计算
P2203 Blink（https://www.luogu.com.cn/problem/P2203）循环节计算
P5550 Chino的数列（https://www.luogu.com.cn/problem/P5550）循环节计算也可以使用矩阵快速幂递推
P7318 「PMOI-4」人赢の梦（https://www.luogu.com.cn/problem/P7318）二维元素，再增加虚拟开始状态，进行循环节计算
P7681 [COCI2008-2009#5] LUBENICA（https://www.luogu.com.cn/problem/P7681）带前缀和的循环节，注意定义循环状态
P1468 [USACO2.2]派对灯 Party Lamps（https://www.luogu.com.cn/problem/P1468）状态压缩求循环节

================================CodeForces================================
C. Yet Another Counting Problem（https://codeforces.com/problemset/problem/1342/C）循环节计数


参考：OI WiKi（xx）
"""


class CircleSection:
    def __init__(self):
        return

    @staticmethod
    def compute_circle_result(n: int, m: int, x: int, tm: int) -> int:

        # 模板: 使用哈希与列表模拟记录循环节开始位置
        dct = dict()
        # 计算 x 每次加 m 加了 tm 次后模 n 的循环态
        lst = []
        while x not in dct:
            dct[x] = len(lst)
            lst.append(x)
            x = (x + m) % n

        # 此时加 m 次数状态为 0.1...length-1
        length = len(lst)
        # 在 ind 次处出现循节
        ind = dct[x]

        # 所求次数不超出循环节
        if tm < length:
            return lst[tm]

        # 所求次数进入循环节
        circle = length - ind
        tm -= length
        j = tm % circle
        return lst[ind + j]

    @staticmethod
    def circle_section_pre(n, grid, c, sta, cur, h):
        # 模板: 需要计算前缀和与循环节
        dct = dict()
        lst = []
        cnt = []
        while sta not in dct:
            dct[sta] = len(dct)
            lst.append(sta)
            cnt.append(c)
            sta = cur
            c = 0
            cur = 0
            for i in range(n):
                num = 1 if sta & (1 << i) else 2
                for j in range(n):
                    if grid[i][j] == "1":
                        c += num
                        cur ^= (num % 2) * (1 << j)

        # 此时次数状态为 0.1...length-1
        length = len(lst)
        # 在 ind 次处出现循节
        ind = dct[sta]
        pre = [0] * (length + 1)
        for i in range(length):
            pre[i + 1] = pre[i] + cnt[i]

        ans = 0
        # 所求次数不超出循环节
        if h < length:
            return ans + pre[h]

        # 所求次数进入循环节
        circle = length - ind
        circle_cnt = pre[length] - pre[ind]

        h -= length
        ans += pre[length]

        ans += (h // circle) * circle_cnt

        j = h % circle
        ans += pre[ind + j] - pre[ind]
        return ans


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
    def lg_p1468(ac=FastIO()):
        # 模板：状态压缩求循环节
        n = ac.read_int()
        op1 = (1 << n) - 1
        op2 = sum(1 << i for i in range(0, n, 2))
        op3 = sum(1 << i for i in range(1, n, 2))
        op4 = sum(1 << i for i in range(0, n, 3))
        stack = [[(1 << n) - 1]]
        ans = []
        while stack:
            path = stack.pop()
            for op in [op1, op2, op3, op4]:
                if path[-1] ^ op in path:
                    ans.append(path[:])
                else:
                    stack.append(path + [path[-1] ^ op])
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


class TestGeneral(unittest.TestCase):

    def test_circle_section(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
