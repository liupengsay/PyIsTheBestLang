import unittest

from src.basis.date.template import DateTime


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

        assert dt.unix_to_time(1462451335) == "2016-05-05 20:28:55"
        assert dt.time_to_unix("2016-05-05 20:28:55") == 1462451335

        ans = 0
        for i in range(10000):
            ans += dt.is_leap_year(i)
            assert ans == dt.leap_year_count(i)
        return


if __name__ == '__main__':
    unittest.main()
