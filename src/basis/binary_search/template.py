from typing import Callable


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠左满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_int_right(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠右满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low

    @staticmethod
    def find_float_left(low: float, high: float, check: Callable, error=1e-6) -> float:
        # 模板: 浮点数范围内二分查找, 选择最靠左满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low: float, high: float, check: Callable, error=1e-6) -> float:
        # 模板: 浮点数范围内二分查找, 选择最靠右满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low
