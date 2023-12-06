"""
Algorithm：日期时间
Function：日期时间之间的变化，年月日时分秒与星期信息


=====================================LuoGu======================================
2655（https://www.luogu.com.cn/problem/P2655）指定日期时分秒过了一定秒数后的具体日期时分秒
1167（https://www.luogu.com.cn/problem/P1167#submit）两个日期之间经过的秒数
5407（https://www.luogu.com.cn/problem/P5407）确定两个日期间隔
5440（https://www.luogu.com.cn/problem/P5440）brute_force日期是否合法且为质数


"""
import datetime
from datetime import datetime, timedelta

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2655(ac=FastIO()):
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