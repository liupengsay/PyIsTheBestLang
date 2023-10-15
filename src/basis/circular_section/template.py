import unittest
from typing import List

from src.fast_io import FastIO



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



