import math
from decimal import Decimal


class TernarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_ceil_point_float(fun, left, right, error=1e-9, high_precision=False):
        """the float point at which the upper convex function obtains its maximum value"""
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_point_int(fun, left, right, error=1):
        """the int point at which the upper convex function obtains its maximum value"""
        while left < right - error:
            diff = (right - left) // 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor_point_float(fun, left, right, error=1e-9, high_precision=False):
        """The float point when solving the convex function to obtain the minimum value"""
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor_point_int(fun, left, right, error=1):
        """The int point when solving the convex function to obtain the minimum value"""
        while left < right - error:
            diff = Decimal(right - left) // 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_value_float(fun, left, right, error=1e-9, high_precision=False):
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
                f2 = dist2
            elif dist1 < dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2

    @staticmethod
    def find_floor_value_float(fun, left, right, error=1e-9, high_precision=False):
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
                f2 = dist2
            elif dist1 > dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2


class TriPartPackTriPart:
    def __init__(self):
        return

    @staticmethod
    def find_floor_point_float(target, left_x, right_x, low_y, high_y):
        # Find the smallest coordinate [x, y] to minimize the target of the objective function
        error = 5e-8

        def optimize(y):
            # The loss function
            low_ = left_x
            high_ = right_x
            while low_ < high_ - error:
                diff_ = (high_ - low_) / 3
                mid1_ = low_ + diff_
                mid2_ = low_ + 2 * diff_
                dist1_ = target(mid1_, y)
                dist2_ = target(mid2_, y)
                if dist1_ < dist2_:
                    high_ = mid2_
                elif dist1_ > dist2_:
                    low_ = mid1_
                else:
                    low_ = mid1_
                    high_ = mid2_
            return low_, target(low_, y)

        low = low_y
        high = high_y
        while low < high - error:
            diff = (high - low) / 3
            mid1 = low + diff
            mid2 = low + 2 * diff
            _, dist1 = optimize(mid1)
            _, dist2 = optimize(mid2)
            if dist1 < dist2:
                high = mid2
            elif dist1 > dist2:
                low = mid1
            else:
                low = mid1
                high = mid2
        res_x, r = optimize(low)
        res_y = low
        return [res_x, res_y, math.sqrt(r)]
