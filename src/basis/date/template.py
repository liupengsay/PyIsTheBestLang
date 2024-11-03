import time
from datetime import datetime, timedelta, date


class DateTime:
    def __init__(self):
        self.leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return

    @staticmethod
    def day_interval(y1, m1, d1, y2, m2, d2):
        """the number of days between two date"""
        day1 = datetime(y1, m1, d1)
        day2 = datetime(y2, m2, d2)
        return (day1 - day2).days

    @staticmethod
    def time_to_unix(dt):
        time_array = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        # for example dt = 2016-05-05 20:28:54
        timestamp = time.mktime(time_array)
        return timestamp

    @staticmethod
    def unix_to_time(timestamp):
        time_local = time.localtime(timestamp)
        # for example timestamp = 1462451334
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        return dt

    def is_leap_year(self, yy):
        assert sum(self.leap_month) == 366  # for example 2000
        assert sum(self.not_leap_month) == 365  # for example 2001
        return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)

    @staticmethod
    def get_n_days(yy, mm, dd, n):
        """the day of after n days from yy-mm-dd"""
        now = datetime(yy, mm, dd, 0, 0, 0, 0)
        delta = timedelta(days=n)
        n_days = now + delta
        return n_days.strftime("%Y-%m-%d")

    @staticmethod
    def is_valid_date(date_str):
        try:
            date.fromisoformat(date_str)
        except ValueError as _:
            return False
        else:
            return True

    def all_palindrome_date(self):
        """brute all the palindrome date from 1000-01-01 to 9999-12-31"""
        ans = []
        for y in range(1000, 10000):
            yy = str(y)
            mm = str(y)[::-1][:2]
            dd = str(y)[::-1][2:]
            if self.is_valid_date(f"{yy}-{mm}-{dd}"):
                ans.append(f"{yy}-{mm}-{dd}")
        return ans

    def unix_minute(self, s):
        """minutes start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = day * 24 * 60 + h * 60 + minute
        return res

    def unix_day(self, s):
        """days start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = day * 24 * 60 + h * 60 + minute
        return res // (24 * 60)

    def unix_second(self, s):
        """seconds start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute, sec = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = (day * 24 * 60 + h * 60 + minute) * 60 + sec
        return res

    @staticmethod
    def leap_year_count(y):
        """leap years count small or equal to y"""
        return 1 + y // 4 - y // 100 + y // 400

    @staticmethod
    def get_start_date(y, m, d, hh, mm, ss, x):
        """the time after any seconds"""
        start_date = datetime(year=y, month=m, day=d, hour=hh, minute=mm, second=ss)
        end_date = start_date + timedelta(seconds=x)
        ans = [end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second]
        return ans
