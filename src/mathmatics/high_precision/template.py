import math
import sys
from decimal import Decimal, getcontext, MAX_PREC
from typing import List

getcontext().prec = MAX_PREC
# sys.set_int_max_str_digits(0)  # important in leetcode big number!


class HighPrecision:
    def __init__(self):
        return

    @staticmethod
    def factorial_to_factorial(n):
        """Compute number of suffixes 0 with 1!*2!***n!"""
        ans = 0
        num = 5
        while num <= n:
            ans += num * (n // num) * (n // num - 1) // 2
            ans += (n // num) * (n % num + 1)
            num *= 5
        return ans

    @staticmethod
    def factorial_to_zero(n):
        """Compute number of suffixes 0 with n!"""
        ans = 0
        while n:
            ans += n // 5
            n //= 5
        return ans

    @staticmethod
    def float_pow(r, n):
        """High precision calculation of the power of decimals"""
        ans = (Decimal(r) ** int(n)).normalize()
        ans = "{:f}".format(ans)
        return ans

    @staticmethod
    def fraction_to_decimal(numerator: int, denominator: int) -> str:
        """Convert fractions to rational numbers or infinitely recurring decimals"""
        if numerator % denominator == 0:
            return str(numerator // denominator) + ".0"
        ans = []
        if numerator * denominator < 0:
            ans.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)

        ans.append(str(numerator // denominator))
        numerator %= denominator
        ans.append(".")
        reminder = numerator % denominator
        dct = dict()
        while reminder and reminder not in dct:
            dct[reminder] = len(ans)
            reminder *= 10
            ans.append(str(reminder // denominator))
            reminder %= denominator
        if reminder in dct:
            ans.insert(dct[reminder], "(")
            ans.append(")")
        return "".join(ans)

    @staticmethod
    def decimal_to_fraction(st):
        # Decimal to Fraction
        def sum_fraction(tmp):
            mu = tmp[0][1]
            for ls in tmp[1:]:
                mu = math.lcm(mu, ls[1])
            zi = sum(ls[0] * mu // ls[1] for ls in tmp)
            mz = math.gcd(mu, zi)
            return [zi // mz, mu // mz]

        # Converting rational numbers or infinite recurring decimals to fractions
        if "." in st:
            lst = st.split(".")
            integer = [int(lst[0]), 1] if lst[0] else [0, 1]
            if "(" not in lst[1]:
                non_repeat = [int(lst[1]), 10 ** len(lst[1])
                              ] if lst[1] else [0, 1]
                repeat = [0, 1]
            else:
                pre, post = lst[1].split("(")
                non_repeat = [int(pre), 10 ** len(pre)] if pre else [0, 1]
                post = post[:-1]
                repeat = [int(post), int("9" * len(post)) * 10 ** len(pre)]
        else:

            integer = [int(st), 1]
            non_repeat = [0, 1]
            repeat = [0, 1]
        return sum_fraction([integer, non_repeat, repeat])


class FloatToFrac:
    def __init__(self):
        # Convert floating-point operations to fractional calculations
        return

    @staticmethod
    def frac_add(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        a = math.lcm(a1, a2)
        b = b1 * (a // a1) + b2 * (a // a2)
        g = math.gcd(b, a)
        b //= g
        a //= g
        if a < 0:
            a *= -1
            b *= -1
        return [b, a]

    @staticmethod
    def frac_max(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 < b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_min(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 > b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_ceil(frac: List[int]) -> int:
        # Both denominators a1 and a2 are required to be non-zero
        b, a = frac
        return math.ceil(b / a)
