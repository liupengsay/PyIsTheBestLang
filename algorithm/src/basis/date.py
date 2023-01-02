"""

"""
"""
算法：xxx
功能：xxx
题目：
Lxxxx xxxx（https://leetcode.cn/problems/shortest-palindrome/）xxxx

参考：OI WiKi（xx）
"""

import datetime
import unittest


class DateTime:
    def __init__(self):
        return

    @staticmethod
    def get_n_days(yy, mm, dd, n):
        # 获取当前日期往后天数的日期
        now = datetime.datetime(yy, mm, dd, 0, 0, 0, 0)
        delta = datetime.timedelta(days=n)
        n_days = now + delta
        return n_days.strftime("%Y-%m-%d")


class TestGeneral(unittest.TestCase):

    def test_date_time(self):
        dt = DateTime()
        assert dt.get_n_days(2023, 1, 2, 1) == "2023-01-03"
        return


if __name__ == '__main__':
    unittest.main()
