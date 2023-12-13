"""
Algorithm：date
Description：date|year|week|month|day|hour|second


=====================================LuoGu======================================
P2655（https://www.luogu.com.cn/problem/P2655）after_date
P1167#submit（https://www.luogu.com.cn/problem/P1167#submit）between_date
P5407（https://www.luogu.com.cn/problem/P5407）between_date
P5440（https://www.luogu.com.cn/problem/P5440）brute_force|prime


"""
import datetime
from datetime import datetime, timedelta

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2655(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2655
        tag: after_date
        """
        # 给定时间起点一定秒数后的具体时间点
        n = ac.read_int()
        for _ in range(n):
            lst = ac.read_list_ints()
            x = (1 << (lst[0] - 1)) - 1
            y = lst[1]
            m, d, h, m, s = lst[2:]
            start_date = datetime(year=y, month=m, day=d, hour=h, minute=m, second=s)
            end_date = start_date + timedelta(seconds=x)
            ac.lst([end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second])
        return
