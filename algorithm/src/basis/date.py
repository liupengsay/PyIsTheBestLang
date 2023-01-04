"""
算法：xxx
功能：xxx
题目：
Lxxxx xxxx（https://leetcode.cn/problems/shortest-palindrome/）xxxx

参考：OI WiKi（xx）
"""
import datetime
import unittest
import time


class DateTime:
    def __init__(self):
        self.leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return

    @staticmethod
    def time_to_unix(dt):
        # 转换成时间 "2016-05-05 20:28:54"
        time_array = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        timestamp = time.mktime(time_array)
        return timestamp

    @staticmethod
    def unix_to_time(timestamp):
        # 转换成时间 1462451334
        time_local = time.localtime(timestamp)
        # 转换成新的时间格式(2016-05-05 20:28:54)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        return dt

    def is_leap_year(self, yy):
        # 判断是否为闰年
        # 闰年天数
        assert sum(self.leap_month) == 366
        assert sum(self.not_leap_month) == 365
        return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)

    @staticmethod
    def get_n_days(yy, mm, dd, n):
        # 获取当前日期往后天数的日期
        now = datetime.datetime(yy, mm, dd, 0, 0, 0, 0)
        delta = datetime.timedelta(days=n)
        n_days = now + delta
        return n_days.strftime("%Y-%m-%d")

    @staticmethod
    def is_valid_date(date_str):
        # 判断日期是否合法
        try:
            datetime.date.fromisoformat(date_str)
        except ValueError as _:
            return False
        else:
            return True

    def all_palidrome_date(self):
        # 枚举出所有的八位回文日期 1000-01-01到9999-12-31
        ans = []
        for y in range(1000, 10000):
            yy = str(y)
            mm = str(y)[::-1][:2]
            dd = str(y)[::-1][2:]
            if self.is_valid_date(f"{yy}-{mm}-{dd}"):
                ans.append(f"{yy}-{mm}-{dd}")
        return ans

    def unix_minute(self, s):
        # 0000-00-00-00:00 开始的分钟数
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute = [int(w) for w in lst[-1].split(":")]
        day = d + 365*y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = day * 24 * 60 + h * 60 + minute
        return res

    @staticmethod
    def leap_year_count(y):
        # 小于等于 y 的闰年数（容斥原理）
        return 1 + y//4 - y//100 + y//400


class TestGeneral(unittest.TestCase):

    def test_date_time(self):
        dt = DateTime()
        assert dt.get_n_days(2023, 1, 2, 1) == "2023-01-03"

        assert dt.is_valid_date("2023-02-29") is False
        assert dt.is_valid_date("2023-02-28") is True
        assert dt.is_valid_date("0001-02-27") is True

        res = dt.all_palidrome_date()
        assert len(res) == 331

        assert dt.is_leap_year(2000) is True
        assert dt.is_leap_year(2100) is False
        assert dt.is_leap_year(0) is True

        assert dt.unix_to_time(1462451334) == "2016-05-05 20:28:54"
        assert dt.time_to_unix("2016-05-05 20:28:54") == 1462451334

        ans = 0
        for i in range(10000):
            ans += dt.is_leap_year(i)
            assert ans == dt.leap_year_count(i)
        return


if __name__ == '__main__':
    unittest.main()
