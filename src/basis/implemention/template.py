from typing import List


class SpiralMatrix:
    def __init__(self):
        return

    @staticmethod
    def joseph_ring(n, m):
        """the last rest for remove the m-th every time in [0,1,...,n-1]"""
        f = 0
        for x in range(2, n + 1):
            f = (m + f) % x
        return f

    @staticmethod
    def num_to_loc(m, n, num):
        """matrix pos from num to loc"""
        # 0123
        # 4567
        m += 1
        return [num // n, num % n]

    @staticmethod
    def loc_to_num(r, c, m, n):
        """matrix pos from loc to num"""
        c += m
        return r * n + n

    @staticmethod
    def get_spiral_matrix_num1(m, n, r, c) -> int:
        """clockwise spiral num at pos [r, c] start from 1"""
        assert 1 <= r <= m and 1 <= c <= n
        num = 1
        while r not in [1, m] and c not in [1, n]:
            num += 2 * m + 2 * n - 4
            r -= 1
            c -= 1
            n -= 2
            m -= 2

        # time complexity is O(m+n)
        x = y = 1
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        while [x, y] != [r, c]:
            a, b = directions[d]
            if not (1 <= x + a <= m and 1 <= y + b <= n):
                d += 1
                a, b = directions[d]
            x += a
            y += b
            num += 1
        return num

    @staticmethod
    def get_spiral_matrix_num2(m, n, r, c) -> int:

        """clockwise spiral num at pos [r, c] start from 1"""
        assert 1 <= r <= m and 1 <= c <= n

        rem = min(r - 1, m - r, c - 1, n - c)
        num = 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        m -= 2 * rem
        n -= 2 * rem
        r -= rem
        c -= rem

        # time complexity is O(1)
        if r == 1:
            num += c
        elif 1 < r <= m and c == n:
            num += n + (r - 1)
        elif r == m and 1 <= c <= n - 1:
            num += n + (m - 1) + (n - c)
        else:
            num += n + (m - 1) + (n - 1) + (m - r)
        return num

    @staticmethod
    def get_spiral_matrix_loc(m, n, num) -> List[int]:
        """clockwise spiral pos of num start from 1"""
        assert 1 <= num <= m * n

        def check(x):
            res = 2 * x * (m - x + 1) + 2 * x * (n - x + 1) - 4 * x
            return res < num

        low = 0
        high = max(m // 2, n // 2)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        rem = high if check(high) else low

        num -= 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        assert num > 0
        m -= 2 * rem
        n -= 2 * rem
        r = c = rem

        if num <= n:
            a = 1
            b = num
        elif n < num <= n + m - 1:
            a = num - n + 1
            b = n
        elif n + (m - 1) < num <= n + (m - 1) + (n - 1):
            a = m
            b = n - (num - n - (m - 1))
        else:
            a = m - (num - n - (n - 1) - (m - 1))
            b = 1
        return [r + a, c + b]
